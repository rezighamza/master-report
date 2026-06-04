"""
CVT-island genetic algorithm for GRADIENT-FREE recovery of a low-rank compressed
model, via per-layer RESIDUAL-RANK ALLOCATION (knowledge distillation, no grads).

Idea:
  The compressed weight W_base[ℓ] = truncate_{k_ℓ}(W_full[ℓ]) discards the tail of
  the spectrum. The optimal rank-r correction to add back is exactly the next r
  singular triplets of the residual (W_full − W_base) — Frobenius-optimal, free to
  compute. So we DON'T evolve the adapter entries (SVD already solves that); we
  evolve HOW MUCH residual rank each layer gets, under a fixed total rank budget.

  Genome:   ranks = [r_0, ..., r_{L-1}]   (per-layer residual rank, 0..rank_cap)
  Student:  W_student[ℓ] = W_base[ℓ] + Σ_{i<r_ℓ} s_i u_i v_iᵀ   (residual top-r_ℓ)
  Budget:   Σ_ℓ r_ℓ·(m_ℓ+n_ℓ) ≤ target_avg_rank · Σ_ℓ (m_ℓ+n_ℓ)
  Fitness:  KL(dense_teacher ‖ student) on a calibration minibatch

The CVT islands diversify WHICH layer-groups (early/late/attn/mlp/down) receive
the rank; an EXP3 bandit picks among rank-reallocation operators; a free
marginal-energy signal (residual singular values) guides energy-greedy swaps.

Usage:
  python enhanced_lowrank_finetune.py \
      --model_name_or_path EleutherAI/pythia-1.4b \
      --lowrank_db ./lowrank_db/pythia/pythia-1.4b/lowrank_svdllm \
      --base_config ./lowrank_db/pythia/pythia-1.4b/lowrank_svdllm/enhanced-lowrank-config.txt \
      --rank_cap 32 --target_avg_rank 8 \
      --calib_split test --calibration_data wikitext2 \
      --calibration_tokens 16384 --calibration_sequence_length 2048 \
      --eval_every 5 --eval_datasets wikitext2 --eval_tokens 524288 --eval_sequence_length 2048 \
      --generations 60 --offspring 16 --survivors_per_selection 4 1 --tokens_per_selection 2048 16384 \
      --num_islands 3 --pop_size 5 --migration_every 3 --stag_threshold 5 --restart_every 25 \
      --fitness_fn kl --use_fast_tokenizer --dtype float16 \
      --log_file ./results/enhanced_lowrank_ft.log
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import random
from tqdm import trange
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.metrics import compute_perplexity, compute_kl_div
from src.model_utils import layer_order_fn
from src.lowrank_utils import load_meta
from src.search_utils import (
    EXP3Bandit, HallOfFame, BasinSpec, Island,
    derive_basins_from_samples, in_basin, sample_minibatch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Genome = {"__id": int, "ranks": [r_0..r_{L-1}]}.  ranks is a LIST so the HoF's
# shallow clone copies it; sig is the exact allocation tuple (for dedup).
# ─────────────────────────────────────────────────────────────────────────────

_GID = [0]
def _new_id() -> int:
    _GID[0] += 1
    return _GID[0]

def make_genome(ranks: List[int]) -> Dict[str, Any]:
    return {"__id": _new_id(), "ranks": list(ranks)}

def clone_genome(g) -> Dict[str, Any]:
    return {"__id": _new_id(), "ranks": list(g["ranks"])}

def genome_sig(g) -> Tuple[int, ...]:
    return tuple(g["ranks"])


# ─────────────────────────────────────────────────────────────────────────────
# Layer classification (attn/mlp, block id, role) from names
# ─────────────────────────────────────────────────────────────────────────────

def classify(layer_names):
    attn, mlp, block_ids, role = set(), set(), [], []
    for name in layer_names:
        n = name.lower()
        if any(k in n for k in ("self_attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "query_key_value")):
            attn.add(name)
        elif any(k in n for k in ("mlp", "gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h")):
            mlp.add(name)
        if any(k in n for k in ("down_proj", "dense_4h_to_h")):
            role.append("down")
        elif any(k in n for k in ("gate_proj", "up_proj", "dense_h_to_4h")):
            role.append("up")
        else:
            role.append("other")
        b = 0
        for part in name.split("."):
            if part.isdigit():
                b = int(part); break
        block_ids.append(b)
    return attn, mlp, block_ids, role


# ─────────────────────────────────────────────────────────────────────────────
# Residual factor bank: per layer, store L_A = U·√S and L_B = √S·Vᵀ truncated to
# rank_cap, so delta_r = L_A[:, :r] @ L_B[:r, :] is the residual top-r correction.
# S_res holds the singular values (free marginal-energy signal).
# ─────────────────────────────────────────────────────────────────────────────

def build_residual_bank(W_full, W_base, layer_names, rank_cap, device):
    LA, LB, S_res, cost = {}, {}, {}, {}
    for name in layer_names:
        res = (W_full[name].float() - W_base[name].float()).to(device)
        m, n = res.shape
        U, S, Vh = torch.linalg.svd(res, full_matrices=False)
        r = min(rank_cap, S.shape[0])
        sq = S[:r].clamp_min(0.0).sqrt()
        LA[name] = (U[:, :r] * sq.unsqueeze(0)).to(torch.float16).contiguous()      # (m, r)
        LB[name] = (sq.unsqueeze(1) * Vh[:r, :]).to(torch.float16).contiguous()      # (r, n)
        S_res[name] = S[:r].float().cpu()
        cost[name] = m + n
        del res, U, S, Vh
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return LA, LB, S_res, cost


def apply_ranks(model, genome, W_base, LA, LB, layer_names):
    """Set each layer's weight to W_base + residual top-r correction. Diff-cached
    against model._cur_ranks so unchanged layers are skipped."""
    ranks = genome["ranks"]
    cur = getattr(model, "_cur_ranks", None)
    if cur is None:
        cur = [-1] * len(layer_names)
    for i, name in enumerate(layer_names):
        r = ranks[i]
        if r == cur[i]:
            continue
        layer = model.get_submodule(name)
        if r <= 0:
            layer.weight.data = W_base[name]
        else:
            delta = LA[name][:, :r] @ LB[name][:r, :]
            layer.weight.data = W_base[name] + delta
    model._cur_ranks = list(ranks)


# ─────────────────────────────────────────────────────────────────────────────
# Budget / allocation helpers
# ─────────────────────────────────────────────────────────────────────────────

def alloc_cost(ranks, cost_list):
    return sum(r * c for r, c in zip(ranks, cost_list))


def repair_to_budget(ranks, cost_list, budget, pref_idx=None):
    """Greedily decrement (weighted to spare pref layers) until under budget."""
    L = len(ranks)
    cost = alloc_cost(ranks, cost_list)
    guard = 0
    while cost > budget and guard < 100000:
        guard += 1
        cand = [i for i in range(L) if ranks[i] > 0]
        if not cand:
            break
        # prefer trimming layers NOT in pref_idx
        if pref_idx is not None:
            non_pref = [i for i in cand if i not in pref_idx]
            if non_pref and random.random() < 0.8:
                cand = non_pref
        i = random.choice(cand)
        ranks[i] -= 1
        cost -= cost_list[i]
    return ranks


def alloc_energy_greedy(S_res, layer_names, cost_list, rank_cap, budget):
    """Static reference + 'energy' seed: fill budget by largest marginal energy s_i²."""
    L = len(layer_names)
    ranks = [0] * L
    cost = 0
    # marginal gain of going r->r+1 at layer i is S_res[i][r]**2
    import heapq
    heap = []
    for i, name in enumerate(layer_names):
        s = S_res[name]
        if s.shape[0] > 0:
            heapq.heappush(heap, (-(float(s[0]) ** 2), i))
    while heap:
        neg_gain, i = heapq.heappop(heap)
        if ranks[i] >= rank_cap:
            continue
        if cost + cost_list[i] > budget:
            continue
        ranks[i] += 1
        cost += cost_list[i]
        name = layer_names[i]
        s = S_res[name]
        if ranks[i] < s.shape[0]:
            heapq.heappush(heap, (-(float(s[ranks[i]]) ** 2), i))
    return ranks


def random_alloc(layer_names, cost_list, rank_cap, budget, bias, attn_set, mlp_set,
                 block_ids, role, S_res):
    """Start from rank_cap everywhere, decrease to budget, sparing 'bias' layers."""
    L = len(layer_names)
    if bias == "energy":
        return alloc_energy_greedy(S_res, layer_names, cost_list, rank_cap, budget)
    n_blocks = max(block_ids) + 1 if block_ids else 1
    half = n_blocks // 2
    def favored(i):
        nm = layer_names[i]
        if bias == "attn":  return nm in attn_set
        if bias == "mlp":   return nm in mlp_set
        if bias == "early": return block_ids[i] < half
        if bias == "late":  return block_ids[i] >= half
        if bias == "down":  return role[i] == "down"
        return False
    ranks = [rank_cap] * L
    cost = alloc_cost(ranks, cost_list)
    guard = 0
    while cost > budget and guard < 200000:
        guard += 1
        cand = [i for i in range(L) if ranks[i] > 0]
        if not cand:
            break
        if bias != "uniform":
            non_fav = [i for i in cand if not favored(i)]
            if non_fav and random.random() < 0.85:
                cand = non_fav
        i = random.choice(cand)
        ranks[i] -= 1
        cost -= cost_list[i]
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Descriptors (6D) — describe the rank-allocation STYLE (cost-weighted shares)
#   0 early, 1 late, 2 attn, 3 down, 4 rank_sparsity, 5 avg_rank/cap
# ─────────────────────────────────────────────────────────────────────────────

def make_descriptor_fn(layer_names, cost_list, rank_cap, attn_set, role, block_ids):
    L = len(layer_names)
    n_blocks = max(block_ids) + 1 if block_ids else 1
    q1 = max(1, n_blocks // 4)
    q4 = n_blocks - q1
    early_idx = [i for i in range(L) if block_ids[i] < q1]
    late_idx = [i for i in range(L) if block_ids[i] >= q4]
    attn_idx = [i for i in range(L) if layer_names[i] in attn_set]
    down_idx = [i for i in range(L) if role[i] == "down"]

    def desc(g):
        ranks = g["ranks"]
        spend = [ranks[i] * cost_list[i] for i in range(L)]
        total = float(sum(spend)) + 1e-12
        early = sum(spend[i] for i in early_idx) / total
        late = sum(spend[i] for i in late_idx) / total
        attn = sum(spend[i] for i in attn_idx) / total
        down = sum(spend[i] for i in down_idx) / total
        sparsity = float(np.mean([1.0 if r == 0 else 0.0 for r in ranks]))
        avg_rank = float(np.mean(ranks)) / max(rank_cap, 1)
        return (
            float(np.clip(early, 0, 1)), float(np.clip(late, 0, 1)),
            float(np.clip(attn, 0, 1)), float(np.clip(down, 0, 1)),
            float(np.clip(sparsity, 0, 1)), float(np.clip(avg_rank, 0, 1)),
        )
    return desc


# ─────────────────────────────────────────────────────────────────────────────
# Operators (rank-reallocation mutations) — all repair back to budget
# ─────────────────────────────────────────────────────────────────────────────

def _pref_indices(ctx, layer_pref):
    if not layer_pref:
        return None
    name_to_idx = ctx["name_to_idx"]
    return set(name_to_idx[n] for n in layer_pref if n in name_to_idx)


def op_realloc(g, ctx, layer_pref=None):
    """Move rank between random layers (a few +1/-1 pairs), then repair."""
    g2 = clone_genome(g)
    ranks = g2["ranks"]
    L = len(ranks)
    cap = ctx["rank_cap"]
    pref = _pref_indices(ctx, layer_pref)
    pool = list(pref) if pref else list(range(L))
    k = random.randint(1, 3)
    for _ in range(k):
        up = [i for i in pool if ranks[i] < cap]
        dn = [i for i in range(L) if ranks[i] > 0]
        if not up or not dn:
            break
        ranks[random.choice(up)] += 1
        ranks[random.choice(dn)] -= 1
    repair_to_budget(ranks, ctx["cost_list"], ctx["budget"], pref)
    return g2


def op_energy_swap(g, ctx, layer_pref=None):
    """Upgrade the highest marginal-energy layer, downgrade the lowest-loss layer."""
    g2 = clone_genome(g)
    ranks = g2["ranks"]
    L = len(ranks)
    cap = ctx["rank_cap"]
    S_res = ctx["S_res"]
    names = ctx["layer_names"]
    pref = _pref_indices(ctx, layer_pref)
    up_pool = list(pref) if pref else list(range(L))
    # highest gain s[r]^2 among layers that can still grow
    best_up, best_up_v = None, -1.0
    for i in up_pool:
        if ranks[i] >= cap:
            continue
        s = S_res[names[i]]
        if ranks[i] < s.shape[0]:
            v = float(s[ranks[i]]) ** 2
            if v > best_up_v:
                best_up_v, best_up = v, i
    # lowest loss s[r-1]^2 among layers that can shrink
    best_dn, best_dn_v = None, float("inf")
    for i in range(L):
        if ranks[i] <= 0:
            continue
        s = S_res[names[i]]
        v = float(s[ranks[i] - 1]) ** 2 if ranks[i] - 1 < s.shape[0] else 0.0
        if v < best_dn_v:
            best_dn_v, best_dn = v, i
    if best_up is not None:
        ranks[best_up] += 1
    if best_dn is not None and best_dn != best_up:
        ranks[best_dn] -= 1
    repair_to_budget(ranks, ctx["cost_list"], ctx["budget"], pref)
    return g2


def op_grow_pref(g, ctx, layer_pref=None):
    """Grow several preferred-group layers; repair frees budget elsewhere."""
    g2 = clone_genome(g)
    ranks = g2["ranks"]
    cap = ctx["rank_cap"]
    pref = _pref_indices(ctx, layer_pref)
    pool = list(pref) if pref else list(range(len(ranks)))
    for i in random.sample(pool, min(len(pool), random.randint(1, 4))):
        ranks[i] = min(cap, ranks[i] + random.randint(1, 3))
    repair_to_budget(ranks, ctx["cost_list"], ctx["budget"], pref)
    return g2


def op_block_shift(g, ctx, layer_pref=None):
    """Move rank from one transformer block to a neighbouring block."""
    g2 = clone_genome(g)
    ranks = g2["ranks"]
    cap = ctx["rank_cap"]
    block_ids = ctx["block_ids"]
    n_blocks = max(block_ids) + 1
    b = random.randint(0, n_blocks - 1)
    nb = max(0, min(n_blocks - 1, b + random.choice([-1, 1])))
    src = [i for i in range(len(ranks)) if block_ids[i] == b and ranks[i] > 0]
    dst = [i for i in range(len(ranks)) if block_ids[i] == nb and ranks[i] < cap]
    for _ in range(random.randint(1, 3)):
        if not src or not dst:
            break
        ranks[random.choice(src)] -= 1
        ranks[random.choice(dst)] += 1
    repair_to_budget(ranks, ctx["cost_list"], ctx["budget"], None)
    return g2


def op_jitter(g, ctx, layer_pref=None):
    """Small ±1 perturbations across several layers, then repair."""
    g2 = clone_genome(g)
    ranks = g2["ranks"]
    cap = ctx["rank_cap"]
    L = len(ranks)
    for i in random.sample(range(L), min(L, random.randint(2, 6))):
        ranks[i] = max(0, min(cap, ranks[i] + random.choice([-1, 1])))
    repair_to_budget(ranks, ctx["cost_list"], ctx["budget"], None)
    return g2


def op_crossover(g1, g2, ctx, layer_pref=None):
    r1, r2 = g1["ranks"], g2["ranks"]
    child = [r1[i] if random.random() < 0.5 else r2[i] for i in range(len(r1))]
    repair_to_budget(child, ctx["cost_list"], ctx["budget"], None)
    return make_genome(child)


OPERATORS = ["realloc", "energy_swap", "grow_pref", "block_shift", "jitter"]
OP_FNS = {
    "realloc": op_realloc,
    "energy_swap": op_energy_swap,
    "grow_pref": op_grow_pref,
    "block_shift": op_block_shift,
    "jitter": op_jitter,
}


def bandit_for_basin(basin: BasinSpec, gamma: float = 0.07) -> EXP3Bandit:
    b = EXP3Bandit(OPERATORS, gamma=gamma)
    sparsity = basin.descriptor_centroid[4]
    if sparsity >= 0.5:   # concentrated allocation → energy-driven growth
        b.w.update({"energy_swap": 2.5, "grow_pref": 2.0, "realloc": 1.5,
                    "block_shift": 1.0, "jitter": 0.5})
    else:                 # spread allocation → fine reshuffling
        b.w.update({"realloc": 2.5, "jitter": 2.0, "energy_swap": 1.5,
                    "block_shift": 1.0, "grow_pref": 0.5})
    return b


def basin_layer_pref(basin: BasinSpec, layer_names, attn_set, mlp_set, block_ids):
    early, late, attn_s = basin.descriptor_centroid[0], basin.descriptor_centroid[1], basin.descriptor_centroid[2]
    if attn_s >= 0.55:
        return [n for n in layer_names if n in attn_set]
    if attn_s <= 0.25:
        return [n for n in layer_names if n in mlp_set]
    n_blocks = max(block_ids) + 1 if block_ids else 1
    half = n_blocks // 2
    if early >= 0.4:
        return [layer_names[i] for i in range(len(layer_names)) if block_ids[i] < half]
    if late >= 0.4:
        return [layer_names[i] for i in range(len(layer_names)) if block_ids[i] >= half]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────────────────────

def eval_pool(model, pool, W_base, LA, LB, layer_names, calib_data, num_tokens,
              fitness_fn, target_logits):
    mb, lmb = sample_minibatch(calib_data, num_tokens, target_logits if fitness_fn == "kl" else None)
    out = []
    for g in pool:
        apply_ranks(model, g, W_base, LA, LB, layer_names)
        if fitness_fn == "ppl":
            out.append(compute_perplexity(model, mb))
        else:
            out.append(compute_kl_div(model, mb, lmb))
    return out


def island_generation(island, layer_pref, descriptor_fn, ctx, model, W_base, LA, LB, layer_names,
                      calib_data, target_logits, n_offspring, pop_size,
                      survivors_per_selection, tokens_per_selection, fitness_fn, hof, crossover_rate):
    bandit = island.bandit
    basin = island.basin
    population = island.pop
    offspring = []
    op_log = []
    op_counts = {op: 0 for op in OPERATORS + ["crossover"]}

    attempts = 0
    while len(offspring) < n_offspring and attempts < n_offspring * 10:
        attempts += 1
        if len(population) >= 2 and random.random() < crossover_rate:
            p1, p2 = random.sample(population, 2)
            child = op_crossover(p1, p2, ctx, layer_pref)
            op = bandit.sample()
            child = OP_FNS[op](child, ctx, layer_pref)
            op_counts["crossover"] += 1
        else:
            parent = random.choice(population)
            op = bandit.sample()
            child = OP_FNS[op](parent, ctx, layer_pref)
        if not in_basin(descriptor_fn(child), basin):
            op2 = bandit.sample()
            cand = OP_FNS[op2](child, ctx, layer_pref)
            if in_basin(descriptor_fn(cand), basin):
                child, op = cand, op2
        op_log.append((op, len(offspring)))
        offspring.append(child)
        op_counts[op] += 1

    elite_pool = list(population)
    stage_survivors = list(survivors_per_selection[:-1]) + [pop_size]
    n_stages = len(stage_survivors)
    pool = list(offspring)
    final_fitnesses = [float("inf")] * len(pool)
    for stage_i, (n_surv, n_tok) in enumerate(zip(stage_survivors, tokens_per_selection)):
        if stage_i == n_stages - 1:
            pool = pool + elite_pool
        n_surv = min(n_surv, len(pool))
        fits = eval_pool(model, pool, W_base, LA, LB, layer_names, calib_data, n_tok,
                         fitness_fn, target_logits)
        best_ids = list(np.argsort(fits)[:n_surv])
        pool = [pool[i] for i in best_ids]
        final_fitnesses = [fits[i] for i in best_ids]

    for g, f in zip(pool, final_fitnesses):
        hof.try_add(g, f)
    survivor_sigs = {genome_sig(s) for s in pool}
    for op, cidx in op_log:
        if cidx < len(offspring):
            r = 1.0 if genome_sig(offspring[cidx]) in survivor_sigs else 0.0
            bandit.update(op, r)
    return pool, (final_fitnesses[0] if final_fitnesses else float("inf")), op_counts


def warm_restart(island, hof, ctx, layer_pref):
    pop_size = len(island.pop)
    if pop_size < 2:
        return
    new_pop = [clone_genome(p) for p in island.pop]
    # worst slot: fresh biased random allocation; second worst: HoF seed
    bias = random.choice(["uniform", "attn", "mlp", "early", "late", "energy"])
    new_pop[-1] = make_genome(random_alloc(
        ctx["layer_names"], ctx["cost_list"], ctx["rank_cap"], ctx["budget"], bias,
        ctx["attn_set"], ctx["mlp_set"], ctx["block_ids"], ctx["role"], ctx["S_res"]))
    if pop_size >= 3 and len(hof) > 0:
        seed = hof.sample_seeds(1)
        if seed:
            new_pop[-2] = clone_genome(seed[0])
    island.pop = new_pop
    for op in island.bandit.ops:
        island.bandit.w[op] = 0.5 * island.bandit.w[op] + 0.5
    island.stag = 0


# ─────────────────────────────────────────────────────────────────────────────
# Tee for logging
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s:
            try: x.write(d); x.flush()
            except Exception: pass
    def flush(self):
        for x in self.s:
            try: x.flush()
            except Exception: pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True, type=str)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--lowrank_db", required=True, type=str)
    p.add_argument("--base_config", required=True, type=str,
                   help="Low-rank config (layer: level) defining the compressed base to recover.")
    p.add_argument("--rank_cap", type=int, default=32,
                   help="Max residual rank a single layer may receive.")
    p.add_argument("--target_avg_rank", type=float, default=8.0,
                   help="Budget = target_avg_rank · Σ(m+n). Equals uniform-rank cost.")
    p.add_argument("--lora_bank", type=str, default=None,
                   help="Path to a trained-LoRA factor bank saved by lora_distill.py "
                        "(--save_bank). If set, the GA allocates over GRADIENT-TRAINED "
                        "components instead of the residual SVD components.")
    p.add_argument("--calibration_data", required=True, type=str)
    p.add_argument("--calibration_tokens", required=True, type=int)
    p.add_argument("--calibration_sequence_length", default=None, type=int)
    p.add_argument("--calib_split", choices=["train", "test"], default="train")
    p.add_argument("--eval_datasets", nargs="+", type=str, default=["wikitext2"])
    p.add_argument("--eval_every", default=5, type=int)
    p.add_argument("--eval_tokens", default=524288, type=int)
    p.add_argument("--eval_sequence_length", default=None, type=int)
    p.add_argument("--fitness_fn", choices=["ppl", "kl"], default="kl")
    p.add_argument("--generations", required=True, type=int)
    p.add_argument("--offspring", required=True, type=int)
    p.add_argument("--survivors_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--tokens_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--num_islands", type=int, default=3)
    p.add_argument("--cvt_samples", type=int, default=120)
    p.add_argument("--cvt_iters", type=int, default=30)
    p.add_argument("--pop_size", type=int, default=5)
    p.add_argument("--migration_every", type=int, default=3)
    p.add_argument("--restart_every", type=int, default=25)
    p.add_argument("--stag_threshold", type=int, default=5)
    p.add_argument("--crossover_rate", type=float, default=0.40)
    p.add_argument("--hof_capacity", type=int, default=20)
    p.add_argument("--dtype", default="float16", choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--attn_implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--log_wandb", default=False, action="store_true")
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--save_ranks", type=str, default=None,
                   help="Optional path to save the best rank allocation (layer: rank).")
    return p.parse_args()


def load_base_config(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, lvl = line.split(":", 1)
            cfg[name.strip()] = int(lvl.strip())
    return cfg


def main():
    args = parse_args()
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)) or ".", exist_ok=True)
        sys.stdout = _Tee(sys.__stdout__, open(args.log_file, "w", buffering=1))
        print(f"[log] tee to {args.log_file}")

    fix_seed(args.seed)
    if args.log_wandb:
        assert has_wandb
        wandb.init(config=args)
    device = "cuda"
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", low_cpu_mem_usage=True,
        torch_dtype=dtype, attn_implementation=args.attn_implementation)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer)

    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings, 8192)
    calib_data = get_data(args.calibration_data, args.calibration_tokens,
                          args.calibration_sequence_length, tokenizer,
                          train=(args.calib_split == "train"))
    args.eval_sequence_length = args.eval_sequence_length or min(model.config.max_position_embeddings, 8192)
    eval_datasets = [get_data(n, args.eval_tokens, args.eval_sequence_length, tokenizer, train=False)
                     for n in args.eval_datasets]

    # 1) DENSE teacher logits (before any compression)
    target_logits = []
    if args.fitness_fn == "kl":
        print("Computing dense teacher logits...", flush=True)
        for i in trange(len(calib_data), desc="Teacher logits", leave=False):
            with torch.no_grad():
                target_logits.append(model(calib_data[i].to(device)).logits.cpu())

    # 2) Snapshot full weights, then apply the low-rank config → compressed base
    meta = load_meta(args.lowrank_db)
    shapes = {k: tuple(v) for k, v in meta["shapes"].items()}
    base_cfg = load_base_config(args.base_config)
    layer_names = sorted([n for n in base_cfg if n in shapes], key=layer_order_fn)
    print(f"Loading compressed base for {len(layer_names)} layers from {args.base_config}", flush=True)
    W_full = {name: model.get_submodule(name).weight.data.clone() for name in layer_names}
    for name in layer_names:
        lvl = base_cfg[name]
        layer = model.get_submodule(name)
        layer.weight.data = torch.load(
            os.path.join(args.lowrank_db, name, f"{lvl}.pth"), map_location=layer.weight.device
        ).to(layer.weight.dtype)
    W_base = {name: model.get_submodule(name).weight.data.clone() for name in layer_names}

    # 3) Factor bank: either the residual SVD (gradient-free) OR a trained-LoRA
    #    bank saved by lora_distill.py (gradient-trained components).
    if args.lora_bank:
        print(f"Loading trained-LoRA factor bank from {args.lora_bank} ...", flush=True)
        bank = torch.load(args.lora_bank, map_location="cpu")
        bank_cap = int(bank.get("rank_cap", args.rank_cap))
        if bank_cap < args.rank_cap:
            print(f"  bank rank_cap={bank_cap} < requested rank_cap={args.rank_cap}; "
                  f"clamping rank_cap to {bank_cap}", flush=True)
            args.rank_cap = bank_cap
        missing = [n for n in layer_names if n not in bank["LA"]]
        if missing:
            raise RuntimeError(f"Trained bank missing {len(missing)} layers, e.g. {missing[:3]}")
        LA, LB, S_res, cost = {}, {}, {}, {}
        for name in layer_names:
            r = min(args.rank_cap, bank["LA"][name].shape[1])
            LA[name] = bank["LA"][name][:, :r].to(device, dtype=torch.float16).contiguous()
            LB[name] = bank["LB"][name][:r, :].to(device, dtype=torch.float16).contiguous()
            S_res[name] = bank["S"][name][:r].float().cpu()
            m, n = W_base[name].shape
            cost[name] = m + n
        del bank, W_full
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        print(f"Building residual factor bank up to rank {args.rank_cap}...", flush=True)
        LA, LB, S_res, cost = build_residual_bank(W_full, W_base, layer_names, args.rank_cap, device)
        del W_full
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    attn_set, mlp_set, block_ids, role = classify(layer_names)
    cost_list = [cost[name] for name in layer_names]
    name_to_idx = {name: i for i, name in enumerate(layer_names)}
    sum_cost = sum(cost_list)
    budget = args.target_avg_rank * sum_cost
    print(f"L={len(layer_names)} Σ(m+n)={sum_cost/1e6:.2f}M  rank_cap={args.rank_cap} "
          f"target_avg_rank={args.target_avg_rank}  budget={budget/1e6:.2f}M", flush=True)

    ctx = {
        "layer_names": layer_names, "cost_list": cost_list, "name_to_idx": name_to_idx,
        "rank_cap": args.rank_cap, "budget": budget, "S_res": S_res, "device": device,
        "attn_set": attn_set, "mlp_set": mlp_set, "role": role, "block_ids": block_ids,
    }

    descriptor_fn = make_descriptor_fn(layer_names, cost_list, args.rank_cap, attn_set, role, block_ids)

    def bounds_fn(members):
        return {}

    # CVT samples: varied biased allocations
    print(f"Sampling {args.cvt_samples} allocations for CVT...", flush=True)
    biases = ["uniform", "attn", "mlp", "early", "late", "down", "energy"]
    samples = [make_genome(random_alloc(layer_names, cost_list, args.rank_cap, budget,
                                        biases[i % len(biases)], attn_set, mlp_set,
                                        block_ids, role, S_res))
               for i in range(args.cvt_samples)]

    basins = derive_basins_from_samples(samples=samples, descriptor_fn=descriptor_fn,
                                        bounds_fn=bounds_fn, k=args.num_islands,
                                        iters=args.cvt_iters, name_prefix="ft", seed=args.seed)
    print(f"CVT produced {len(basins)} basins:")
    for b in basins:
        print(f"  {b.name} mass={b.descriptor_mass} centroid={tuple(round(c,2) for c in b.descriptor_centroid)} r={b.descriptor_radius:.2f}")

    islands = []
    island_prefs = []
    for basin in basins:
        bandit = bandit_for_basin(basin)
        lpref = basin_layer_pref(basin, layer_names, attn_set, mlp_set, block_ids)
        nearest = sorted(samples, key=lambda g: math.sqrt(sum(
            ((d - c) / max(s, 1e-6)) ** 2
            for d, c, s in zip(descriptor_fn(g), basin.descriptor_centroid, basin.descriptor_scale)
        )))[:args.pop_size]
        pop = [clone_genome(g) for g in nearest]
        while len(pop) < args.pop_size:
            pop.append(make_genome(random_alloc(layer_names, cost_list, args.rank_cap, budget,
                                                "uniform", attn_set, mlp_set, block_ids, role, S_res)))
        islands.append(Island(name=basin.name, basin=basin, pop=pop, bandit=bandit,
                              stag_threshold=args.stag_threshold))
        island_prefs.append(lpref)

    hof = HallOfFame(capacity=args.hof_capacity, sig_fn=genome_sig)

    # References at the SAME budget
    print("Reference PPLs (same budget):")
    baseline_g = make_genome([0] * len(layer_names))   # no correction
    apply_ranks(model, baseline_g, W_base, LA, LB, layer_names)
    for dname, dset in zip(args.eval_datasets, eval_datasets):
        print(f"  [baseline rank0] {dname} ppl={compute_perplexity(model, dset):.3f}", flush=True)
    uni_r = max(0, min(args.rank_cap, int(round(args.target_avg_rank))))
    uniform_g = make_genome([uni_r] * len(layer_names))
    apply_ranks(model, uniform_g, W_base, LA, LB, layer_names)
    for dname, dset in zip(args.eval_datasets, eval_datasets):
        print(f"  [uniform rank{uni_r}] {dname} ppl={compute_perplexity(model, dset):.3f}", flush=True)
    energy_g = make_genome(alloc_energy_greedy(S_res, layer_names, cost_list, args.rank_cap, budget))
    apply_ranks(model, energy_g, W_base, LA, LB, layer_names)
    for dname, dset in zip(args.eval_datasets, eval_datasets):
        print(f"  [energy-greedy] {dname} ppl={compute_perplexity(model, dset):.3f}", flush=True)

    # Seed the search with the energy-greedy allocation (strong static baseline)
    islands[0].pop[-1] = clone_genome(energy_g)
    hof.try_add(energy_g, float("inf"))

    global_best_fitness = float("inf")
    global_best = clone_genome(energy_g)
    log_dict = {}

    print(f"\nRunning {args.generations} gens × {len(islands)} islands.", flush=True)
    for gen in trange(args.generations, desc="Generations"):
        for island, lpref in zip(islands, island_prefs):
            new_pop, best_f, _ = island_generation(
                island, lpref, descriptor_fn, ctx, model, W_base, LA, LB, layer_names,
                calib_data, target_logits if target_logits else None,
                args.offspring, args.pop_size, args.survivors_per_selection,
                args.tokens_per_selection, args.fitness_fn, hof, args.crossover_rate)
            island.pop = new_pop
            island.record(best_f)
            print(f"  [gen {gen}] {island.name} best_f={best_f:.4e} stag={island.stag}", flush=True)
            if best_f < global_best_fitness:
                global_best_fitness = best_f
                global_best = clone_genome(new_pop[0])

        if (gen + 1) % args.migration_every == 0:
            bests = [clone_genome(isl.pop[0]) for isl in islands]
            for i, isl in enumerate(islands):
                others = [bests[j] for j in range(len(islands)) if j != i]
                random.shuffle(others)
                for k in range(min(2, len(isl.pop) - 1, len(others))):
                    isl.pop[-(k + 1)] = others[k]

        for island, lpref in zip(islands, island_prefs):
            if island.is_stagnant():
                warm_restart(island, hof, ctx, lpref)
        if args.restart_every > 0 and (gen + 1) % args.restart_every == 0:
            for island, lpref in zip(islands, island_prefs):
                warm_restart(island, hof, ctx, lpref)

        if gen % args.eval_every == 0:
            apply_ranks(model, global_best, W_base, LA, LB, layer_names)
            avg_rank = float(np.mean(global_best["ranks"]))
            spent = alloc_cost(global_best["ranks"], cost_list)
            for dname, dset in zip(args.eval_datasets, eval_datasets):
                ppl = compute_perplexity(model, dset)
                print(f"  [{dname}] ppl={ppl:.3f}", flush=True)
                log_dict[f"ppl_eval/{dname}"] = ppl
            print(f"  fit={global_best_fitness:.4e}  avg_rank={avg_rank:.2f}  spent={spent/1e6:.2f}M/{budget/1e6:.2f}M", flush=True)
            for island in islands:
                print(f"  [{island.name}] {island.bandit.summary()}", flush=True)
        if args.log_wandb:
            wandb.log(log_dict)

    hof_g, hof_f = hof.best()
    if hof_g is not None and hof_f < global_best_fitness:
        global_best, global_best_fitness = hof_g, hof_f

    apply_ranks(model, global_best, W_base, LA, LB, layer_names)
    print("\nFINAL (best allocation):")
    avg_rank = float(np.mean(global_best["ranks"]))
    print(f"  avg_rank={avg_rank:.2f}  spent={alloc_cost(global_best['ranks'], cost_list)/1e6:.2f}M/{budget/1e6:.2f}M", flush=True)
    for dname, dset in zip(args.eval_datasets, eval_datasets):
        print(f"  [{dname}] ppl={compute_perplexity(model, dset):.3f}", flush=True)

    if args.save_ranks:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_ranks)) or ".", exist_ok=True)
        with open(args.save_ranks, "w") as f:
            for name, r in zip(layer_names, global_best["ranks"]):
                f.write(f"{name}: {r}\n")
        print(f"Saved rank allocation → {args.save_ranks}")


if __name__ == "__main__":
    main()
