"""
Enhanced (CVT island) search for low-rank rank allocation.

Genome:   grouped per-layer rank-LEVEL indices, Tuple[Tuple[int]].
Cost:     sum_layer rank[level]*(m+n)  <=  target_ratio * total_original_params.
Fitness:  KL(low-rank model || dense model).

Mirrors enhanced_quant_search.py (EXP3-IX islands, CVT basins, cross-pollinating
migration, gentle restart, hill-climb) but:
  - cost is realized low-rank params, not bits×numel
  - descriptors use spectral-energy retention (free SVD signal)
  - adds the energy_greedy operator (salience-guided swap from singular values)

Usage:
  python enhanced_lowrank_search.py \
      --model_name_or_path EleutherAI/pythia-1.4b \
      --lowrank_db ./lowrank_db/pythia/pythia-1.4b/lowrank \
      --target_ratio 0.5 \
      --calibration_data fineweb_edu --calibration_tokens 131072 --calibration_sequence_length 2048 \
      --eval_every 5 --eval_datasets fineweb_edu wikitext2 c4 \
      --eval_tokens 524288 --eval_sequence_length 2048 \
      --generations 80 --offspring 32 \
      --survivors_per_selection 4 1 --tokens_per_selection 2048 16384 \
      --num_islands 3 --pop_size 5 --migration_every 3 --stag_threshold 5 --restart_every 25 \
      --fitness_fn kl --use_fast_tokenizer --dtype float16 \
      --log_file ./results/enhanced_lowrank_0.5.log
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from tqdm import trange
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from src.model_utils import layer_order_fn, group_layers
from src.lowrank_utils import load_meta, build_cost_tables, load_layer_ranks, energy_retained
from src.search_utils import (
    EXP3Bandit, HallOfFame, CosineAnnealingTemperature, BasinSpec, Island,
    derive_basins_from_samples, in_basin, sample_minibatch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading
# ─────────────────────────────────────────────────────────────────────────────

def load_layers(model, grouped_names, new_state, db_path):
    assert hasattr(model, "state")
    for gi in range(len(grouped_names)):
        for name, new_lv, old_lv in zip(grouped_names[gi], new_state[gi], model.state[gi]):
            if new_lv != old_lv:
                layer = model.get_submodule(name)
                layer.weight.data = torch.load(
                    os.path.join(db_path, name, f"{new_lv}.pth"),
                    map_location=layer.weight.device,
                ).to(layer.weight.dtype)
    model.state = new_state


def genome_sig(g) -> Tuple:
    return tuple(tuple(row) for row in g)

def clone_genome(g):
    return tuple(tuple(row) for row in g)


# ─────────────────────────────────────────────────────────────────────────────
# Layer classification (attn/mlp, block ids, roles) — same as quant
# ─────────────────────────────────────────────────────────────────────────────

def classify_layers(grouped_names):
    flat, attn, mlp, block_ids, role = [], [], [], [], []
    for gi, group in enumerate(grouped_names):
        for li, name in enumerate(group):
            flat.append((gi, li, name))
            idx = len(flat) - 1
            n = name.lower()
            if "self_attn" in n or "attention" in n or any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "query_key_value")):
                attn.append(idx)
            elif "mlp" in n or any(k in n for k in ("gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h")):
                mlp.append(idx)
            if any(k in n for k in ("query_key_value", "q_proj", "k_proj", "v_proj")):
                role.append("qkv_in")
            elif any(k in n for k in ("o_proj", "attention.dense")) and "h_to_4h" not in n:
                role.append("attn_out")
            elif any(k in n for k in ("gate_proj", "up_proj", "dense_h_to_4h")):
                role.append("up_proj")
            elif any(k in n for k in ("down_proj", "dense_4h_to_h")):
                role.append("down_proj")
            else:
                role.append("other")
            block_idx = 0
            for part in name.split("."):
                if part.isdigit():
                    block_idx = int(part); break
            block_ids.append(block_idx)
    return flat, attn, mlp, block_ids, role


# ─────────────────────────────────────────────────────────────────────────────
# Descriptors (6D) — low-rank specific
# ─────────────────────────────────────────────────────────────────────────────

def make_descriptor_fn(flat_to_group, level_costs_flat, energy_flat, attn_flat, block_ids):
    """
    0: early_rank_share   — cost fraction in first 25% of blocks
    1: late_rank_share    — cost fraction in last 25% of blocks
    2: attn_rank_share    — attention cost fraction
    3: energy_mean        — mean retained spectral energy at chosen levels
    4: energy_spread      — std of per-layer retained energy
    5: longest_high_run   — longest contiguous run of above-mean-level blocks / L
    """
    N = len(flat_to_group)
    n_blocks = max(block_ids) + 1 if block_ids else 1
    q1 = max(1, n_blocks // 4)
    q4 = n_blocks - q1
    attn_set = set(attn_flat)

    def desc(g):
        levels = [g[gi][li] for (gi, li) in flat_to_group]
        costs = [level_costs_flat[i][levels[i]] for i in range(N)]
        energies = [energy_flat[i][levels[i]] for i in range(N)]
        total = sum(costs)
        if total <= 0:
            return (0.25, 0.25, 0.5, 0.5, 0.0, 0.0)
        early = sum(costs[i] for i in range(N) if block_ids[i] < q1) / total
        late = sum(costs[i] for i in range(N) if block_ids[i] >= q4) / total
        attn_s = sum(costs[i] for i in range(N) if i in attn_set) / total
        e_mean = float(np.mean(energies))
        e_spread = float(np.std(energies))
        avg_lv = float(np.mean(levels))
        block_high = [False] * n_blocks
        for i in range(N):
            if levels[i] > avg_lv:
                block_high[block_ids[i]] = True
        longest = cur = 0
        for v in block_high:
            cur = cur + 1 if v else 0
            longest = max(longest, cur)
        return (
            float(np.clip(early, 0, 1)),
            float(np.clip(late, 0, 1)),
            float(np.clip(attn_s, 0, 1)),
            float(np.clip(e_mean, 0, 1)),
            float(np.clip(e_spread, 0, 1)),
            float(np.clip(longest / max(1, n_blocks), 0, 1)),
        )
    return desc


def make_bounds_fn():
    def bounds(members):
        flat = []
        for g in members:
            for row in g:
                flat.extend(row)
        return {"lvl_lo": int(np.quantile(flat, 0.05)) if flat else 0,
                "lvl_hi": int(np.quantile(flat, 0.95)) if flat else 0}
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# Sampling under budget
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_genome(grouped_names, level_costs, n_levels, budget, flat_to_group,
                         attn_flat=None, bias="uniform"):
    """Start at max level, randomly decrease to budget. Bias shifts which layers keep rank."""
    g = [[n_levels - 1 for _ in group] for group in grouped_names]
    N = len(flat_to_group)
    attn_set = set(attn_flat or [])

    def cost():
        return sum(level_costs[grouped_names[gi][li]][g[gi][li]] for (gi, li) in flat_to_group)

    # weights for which layers to decrease first (bias = keep some, shrink others)
    def decr_weight(fi):
        if bias == "attn_heavy":
            return 0.3 if fi in attn_set else 1.0   # protect attn → shrink mlp
        if bias == "mlp_heavy":
            return 1.0 if fi in attn_set else 0.3   # protect mlp → shrink attn
        return 1.0

    c = cost()
    guard = 0
    while c > budget and guard < N * 40:
        guard += 1
        # choose a flat layer to decrease, weighted
        cands = [fi for fi in range(N) if g[flat_to_group[fi][0]][flat_to_group[fi][1]] > 0]
        if not cands:
            break
        w = [decr_weight(fi) for fi in cands]
        fi = random.choices(cands, weights=w)[0]
        gi, li = flat_to_group[fi]
        name = grouped_names[gi][li]
        c -= level_costs[name][g[gi][li]] - level_costs[name][g[gi][li] - 1]
        g[gi][li] -= 1
    return tuple(tuple(r) for r in g)


# ─────────────────────────────────────────────────────────────────────────────
# Operators (budget-preserving level swaps)
# ─────────────────────────────────────────────────────────────────────────────

def _cost_of(g, grouped_names, level_costs, flat_to_group):
    return sum(level_costs[grouped_names[gi][li]][g[gi][li]] for (gi, li) in flat_to_group)


def _pick_flat(N, layer_pref):
    if layer_pref and random.random() < 0.8:
        return random.choice(layer_pref)
    return random.randrange(N)


def op_rank_swap(g, ctx, layer_pref=None):
    grouped_names, level_costs, n_levels, budget, flat_to_group = ctx["names"], ctx["costs"], ctx["n_levels"], ctx["budget"], ctx["f2g"]
    N = len(flat_to_group)
    g2 = [list(r) for r in g]
    for _ in range(400):
        ii = _pick_flat(N, layer_pref)
        gi_i, li_i = flat_to_group[ii]
        if g2[gi_i][li_i] >= n_levels - 1:
            continue
        di = _pick_flat(N, layer_pref)
        if di == ii:
            continue
        gi_d, li_d = flat_to_group[di]
        if g2[gi_d][li_d] <= 0:
            continue
        g2[gi_i][li_i] += 1
        g2[gi_d][li_d] -= 1
        if _cost_of(g2, grouped_names, level_costs, flat_to_group) <= budget:
            return tuple(tuple(r) for r in g2)
        g2[gi_i][li_i] -= 1
        g2[gi_d][li_d] += 1
    return tuple(tuple(r) for r in g2)


def op_rank_walk_2(g, ctx, layer_pref=None):
    return op_rank_swap(op_rank_swap(g, ctx, layer_pref), ctx, layer_pref)

def op_rank_walk_3(g, ctx, layer_pref=None):
    return op_rank_swap(op_rank_walk_2(g, ctx, layer_pref), ctx, layer_pref)


def op_rank_rebalance(g, ctx, layer_pref=None):
    grouped_names, level_costs, flat_to_group = ctx["names"], ctx["costs"], ctx["f2g"]
    N = len(flat_to_group)
    pool = layer_pref if layer_pref else list(range(N))
    if not pool:
        return clone_genome(g)
    g2 = [list(r) for r in g]
    lvls = [(fi, g2[flat_to_group[fi][0]][flat_to_group[fi][1]]) for fi in pool]
    src = max(lvls, key=lambda x: x[1])[0]   # highest level → decrease
    dst = min(lvls, key=lambda x: x[1])[0]   # lowest level → increase
    gi_s, li_s = flat_to_group[src]
    gi_d, li_d = flat_to_group[dst]
    if g2[gi_s][li_s] <= 0 or g2[gi_d][li_d] >= ctx["n_levels"] - 1:
        return op_rank_swap(g, ctx, layer_pref)
    g2[gi_s][li_s] -= 1
    g2[gi_d][li_d] += 1
    if _cost_of(g2, grouped_names, level_costs, flat_to_group) > ctx["budget"]:
        return op_rank_swap(g, ctx, layer_pref)
    return tuple(tuple(r) for r in g2)


def op_block_coord(g, ctx, layer_pref=None):
    grouped_names, level_costs, n_levels, budget, flat_to_group = ctx["names"], ctx["costs"], ctx["n_levels"], ctx["budget"], ctx["f2g"]
    block_ids, role_labels = ctx["block_ids"], ctx["roles"]
    N = len(flat_to_group)
    n_blocks = max(block_ids) + 1
    g2 = [list(r) for r in g]
    for _ in range(50):
        blk = random.randrange(n_blocks)
        side = random.choice(["attn", "mlp"])
        tgt = {"qkv_in", "attn_out"} if side == "attn" else {"up_proj", "down_proj"}
        dirn = random.choice([-1, +1])
        members = [i for i in range(N) if block_ids[i] == blk and role_labels[i] in tgt]
        if not members:
            continue
        changed = []
        for i in members:
            gi, li = flat_to_group[i]
            nv = g2[gi][li] + dirn
            if 0 <= nv <= n_levels - 1:
                g2[gi][li] = nv
                changed.append(i)
        if not changed:
            continue
        c = _cost_of(g2, grouped_names, level_costs, flat_to_group)
        guard = 0
        while c > budget and guard < 200:
            guard += 1
            ri = random.randrange(N)
            gi, li = flat_to_group[ri]
            if g2[gi][li] <= 0:
                continue
            name = grouped_names[gi][li]
            c -= level_costs[name][g2[gi][li]] - level_costs[name][g2[gi][li] - 1]
            g2[gi][li] -= 1
        if c <= budget:
            return tuple(tuple(r) for r in g2)
        for i in changed:
            gi, li = flat_to_group[i]
            g2[gi][li] -= dirn
    return tuple(tuple(r) for r in g2)


def op_energy_greedy(g, ctx, layer_pref=None):
    """
    Low-rank-specific: use the FREE singular-value energy signal.
    Upgrade the layer with the largest marginal energy gain (+1 level),
    downgrade the layer with the smallest marginal energy loss (-1 level).
    Keeps budget. No forward pass needed for the decision.
    """
    grouped_names, level_costs, n_levels, budget, flat_to_group = ctx["names"], ctx["costs"], ctx["n_levels"], ctx["budget"], ctx["f2g"]
    energy_flat = ctx["energy"]
    N = len(flat_to_group)
    g2 = [list(r) for r in g]
    flat_lv = [g2[flat_to_group[i][0]][flat_to_group[i][1]] for i in range(N)]

    # Marginal energy gain for +1 (upgrade), marginal loss for -1 (downgrade)
    up_gain = []
    for i in range(N):
        lv = flat_lv[i]
        if lv < n_levels - 1:
            up_gain.append((energy_flat[i][lv + 1] - energy_flat[i][lv], i))
    down_loss = []
    for i in range(N):
        lv = flat_lv[i]
        if lv > 0:
            down_loss.append((energy_flat[i][lv] - energy_flat[i][lv - 1], i))
    if not up_gain or not down_loss:
        return op_rank_swap(g, ctx, layer_pref)
    # Upgrade the highest-gain layer, downgrade the lowest-loss layer
    _, ui = max(up_gain, key=lambda x: x[0])
    down_loss.sort(key=lambda x: x[0])
    for _, di in down_loss:
        if di == ui:
            continue
        gi_u, li_u = flat_to_group[ui]
        gi_d, li_d = flat_to_group[di]
        g2[gi_u][li_u] += 1
        g2[gi_d][li_d] -= 1
        if _cost_of(g2, grouped_names, level_costs, flat_to_group) <= budget:
            return tuple(tuple(r) for r in g2)
        g2[gi_u][li_u] -= 1
        g2[gi_d][li_d] += 1
    return op_rank_swap(g, ctx, layer_pref)


def op_crossover(g1, g2, ctx, layer_pref=None):
    grouped_names, level_costs, n_levels, budget, flat_to_group = ctx["names"], ctx["costs"], ctx["n_levels"], ctx["budget"], ctx["f2g"]
    child = [list(r) for r in g1]
    for (gi, li) in flat_to_group:
        if random.random() < 0.5:
            child[gi][li] = g2[gi][li]
    # Repair to budget
    c = _cost_of(child, grouped_names, level_costs, flat_to_group)
    guard = 0
    N = len(flat_to_group)
    while c > budget and guard < 1000:
        guard += 1
        fi = random.randrange(N)
        gi, li = flat_to_group[fi]
        if child[gi][li] <= 0:
            continue
        name = grouped_names[gi][li]
        c -= level_costs[name][child[gi][li]] - level_costs[name][child[gi][li] - 1]
        child[gi][li] -= 1
    return tuple(tuple(r) for r in child)


OPERATORS = ["rank_swap", "rank_walk_2", "rank_walk_3", "rank_rebalance", "block_coord", "energy_greedy"]
OP_FNS = {
    "rank_swap": op_rank_swap,
    "rank_walk_2": op_rank_walk_2,
    "rank_walk_3": op_rank_walk_3,
    "rank_rebalance": op_rank_rebalance,
    "block_coord": op_block_coord,
    "energy_greedy": op_energy_greedy,
}


def bandit_for_basin(basin: BasinSpec, gamma: float = 0.07) -> EXP3Bandit:
    b = EXP3Bandit(OPERATORS, gamma=gamma)
    e_spread = basin.descriptor_centroid[4]  # energy spread axis
    if e_spread >= 0.15:   # lopsided energy → fine swaps + energy guidance
        b.w.update({"rank_swap": 2.5, "energy_greedy": 2.5, "rank_walk_2": 1.5,
                    "rank_rebalance": 1.0, "block_coord": 1.0, "rank_walk_3": 0.5})
    else:                  # uniform energy → coarse moves
        b.w.update({"rank_walk_3": 2.5, "rank_walk_2": 2.0, "block_coord": 1.5,
                    "rank_rebalance": 1.5, "rank_swap": 1.0, "energy_greedy": 1.0})
    return b


def basin_layer_pref(basin, attn_flat, mlp_flat, block_ids):
    early, late, attn_s = basin.descriptor_centroid[0], basin.descriptor_centroid[1], basin.descriptor_centroid[2]
    if attn_s >= 0.55:
        return attn_flat
    if attn_s <= 0.30:
        return mlp_flat
    n_blocks = max(block_ids) + 1 if block_ids else 1
    half = n_blocks // 2
    if early >= 0.4:
        return [i for i in range(len(block_ids)) if block_ids[i] < half]
    if late >= 0.4:
        return [i for i in range(len(block_ids)) if block_ids[i] >= half]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────────────────────

def compute_fitness(model, data, fitness_fn, target_logits=None) -> float:
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    return compute_kl_div(model, data, target_logits)


def eval_pool(model, grouped_names, db_path, pool, calib_data, num_tokens, fitness_fn, target_logits):
    mb, lmb = sample_minibatch(calib_data, num_tokens, target_logits if fitness_fn == "kl" else None)
    out = []
    for g in pool:
        load_layers(model, grouped_names, g, db_path)
        out.append(compute_fitness(model, mb, fitness_fn, lmb))
    return out


def hill_climb_elite(elite, ctx, eval_fn):
    grouped_names, level_costs, n_levels, budget, flat_to_group = ctx["names"], ctx["costs"], ctx["n_levels"], ctx["budget"], ctx["f2g"]
    N = len(flat_to_group)
    candidates = []
    for i in range(N):
        gi_i, li_i = flat_to_group[i]
        if elite[gi_i][li_i] >= n_levels - 1:
            continue
        for j in range(N):
            if i == j:
                continue
            gi_j, li_j = flat_to_group[j]
            if elite[gi_j][li_j] <= 0:
                continue
            cand = [list(r) for r in elite]
            cand[gi_i][li_i] += 1
            cand[gi_j][li_j] -= 1
            if _cost_of(cand, grouped_names, level_costs, flat_to_group) > budget:
                continue
            mag = level_costs[grouped_names[gi_i][li_i]][1] + level_costs[grouped_names[gi_j][li_j]][1]
            candidates.append((tuple(tuple(r) for r in cand), mag))
    if not candidates:
        return elite, float("inf")
    candidates.sort(key=lambda x: -x[1])
    best_g, best_f = elite, float("inf")
    for cand_g, _ in candidates[:64]:
        f = eval_fn(cand_g)
        if f < best_f:
            best_f, best_g = f, cand_g
    return best_g, best_f


# ─────────────────────────────────────────────────────────────────────────────
# Island generation
# ─────────────────────────────────────────────────────────────────────────────

def island_generation(island, layer_pref, descriptor_fn, ctx, model, grouped_names, db_path,
                      calib_data, target_logits, n_offspring, pop_size,
                      survivors_per_selection, tokens_per_selection, fitness_fn, hof, crossover_rate):
    bandit = island.bandit
    basin = island.basin
    population = island.pop
    offspring = []
    op_log = []
    seen = {genome_sig(p) for p in population}
    op_counts = {op: 0 for op in OPERATORS + ["crossover"]}

    attempts = 0
    while len(offspring) < n_offspring and attempts < n_offspring * 30:
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
        sig = genome_sig(child)
        if sig in seen:
            continue
        seen.add(sig)
        op_log.append((op, len(offspring)))
        offspring.append(child)
        op_counts[op] += 1

    elite_pool = [p for p in population if genome_sig(p) not in {genome_sig(c) for c in offspring}]
    stage_survivors = list(survivors_per_selection[:-1]) + [pop_size]
    n_stages = len(stage_survivors)
    pool = list(offspring)
    final_fitnesses = [float("inf")] * len(pool)
    for stage_i, (n_surv, n_tok) in enumerate(zip(stage_survivors, tokens_per_selection)):
        if stage_i == n_stages - 1:
            for e in elite_pool:
                if genome_sig(e) not in {genome_sig(p) for p in pool}:
                    pool.append(e)
        n_surv = min(n_surv, len(pool))
        fits = eval_pool(model, grouped_names, db_path, pool, calib_data, n_tok, fitness_fn, target_logits)
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


def warm_restart_island(island, hof, ctx, layer_pref, attn_flat):
    pop_size = len(island.pop)
    if pop_size < 2:
        return
    new_pop = [clone_genome(p) for p in island.pop]
    new_pop[-1] = sample_random_genome(
        ctx["names"], ctx["costs"], ctx["n_levels"], ctx["budget"], ctx["f2g"],
        attn_flat=attn_flat, bias=random.choice(["uniform", "attn_heavy", "mlp_heavy"]))
    if pop_size >= 3 and len(hof) > 0:
        seed = hof.sample_seeds(1)
        if seed:
            new_pop[-2] = clone_genome(seed[0])
    island.pop = new_pop
    for op in island.bandit.ops:
        island.bandit.w[op] = 0.5 * island.bandit.w[op] + 0.5
    island.stag = 0


# ─────────────────────────────────────────────────────────────────────────────
# Args + Main
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams:
            try: s.write(d); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True, type=str)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--calibration_data", required=True, type=str)
    p.add_argument("--calibration_tokens", required=True, type=int)
    p.add_argument("--calibration_sequence_length", default=None, type=int)
    p.add_argument("--eval_datasets", nargs="+", type=str, default=["fineweb_edu", "wikitext2", "c4"])
    p.add_argument("--eval_every", default=5, type=int)
    p.add_argument("--eval_tokens", default=524288, type=int)
    p.add_argument("--eval_sequence_length", default=None, type=int)
    p.add_argument("--fitness_fn", choices=["ppl", "kl"], default="kl")
    p.add_argument("--calib_split", choices=["train", "test"], default="train",
                   help="Calibration split. Use 'test' locally on Windows to avoid the "
                        "train-split giant-string tokenization crash (note: overlaps eval).")
    p.add_argument("--lowrank_db", required=True, type=str)
    p.add_argument("--target_ratio", required=True, type=float)
    p.add_argument("--group_rule", choices=["none", "name", "size"], default="size")
    p.add_argument("--generations", required=True, type=int)
    p.add_argument("--offspring", required=True, type=int)
    p.add_argument("--survivors_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--tokens_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--num_islands", type=int, default=3)
    p.add_argument("--cvt_samples", type=int, default=300)
    p.add_argument("--cvt_iters", type=int, default=30)
    p.add_argument("--pop_size", type=int, default=5)
    p.add_argument("--migration_every", type=int, default=3)
    p.add_argument("--restart_every", type=int, default=25)
    p.add_argument("--stag_threshold", type=int, default=5)
    p.add_argument("--crossover_rate", type=float, default=0.40)
    p.add_argument("--hof_capacity", type=int, default=30)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--attn_implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--log_wandb", default=False, action="store_true")
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--configuration_name", type=str, default="enhanced-lowrank-config.txt")
    return p.parse_args()


def main():
    args = parse_args()
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)) or ".", exist_ok=True)
        sys.stdout = _Tee(sys.__stdout__, open(args.log_file, "w", buffering=1))
        print(f"[log] tee to {args.log_file}")

    assert len(args.survivors_per_selection) == len(args.tokens_per_selection)
    fix_seed(args.seed)
    if args.log_wandb:
        assert has_wandb
        wandb.init(config=args)
    device = "cuda"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", low_cpu_mem_usage=True,
        torch_dtype=args.dtype, attn_implementation=args.attn_implementation)
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

    target_logits = []
    if args.fitness_fn == "kl":
        for i in trange(len(calib_data), desc="Target logits", leave=False):
            with torch.no_grad():
                target_logits.append(model(calib_data[i].to(device)).logits.cpu())

    # Layers, grouping, costs, energies
    meta = load_meta(args.lowrank_db)
    level_costs, orig_costs = build_cost_tables(args.lowrank_db, meta)
    layer_names = sorted(meta["layer_names"], key=layer_order_fn)
    grouped_names = group_layers(model, layer_names, args.group_rule)
    model.state = [[None] * len(g) for g in grouped_names]
    n_levels = meta["num_levels"]
    total_orig = sum(orig_costs.values())
    budget = args.target_ratio * total_orig
    print(f"Levels={n_levels} total_orig={total_orig/1e6:.1f}M budget={budget/1e6:.1f}M (ratio {args.target_ratio})")

    flat_to_group = [(gi, li) for gi, group in enumerate(grouped_names) for li in range(len(group))]
    _flat, attn_flat, mlp_flat, block_ids, role_labels = classify_layers(grouped_names)

    # Flat cost & energy tables
    level_costs_flat = [level_costs[grouped_names[gi][li]] for (gi, li) in flat_to_group]
    energy_flat = []
    for (gi, li) in flat_to_group:
        name = grouped_names[gi][li]
        energy_flat.append(energy_retained(args.lowrank_db, name, load_layer_ranks(args.lowrank_db, name)))

    ctx = {
        "names": grouped_names, "costs": level_costs, "n_levels": n_levels,
        "budget": budget, "f2g": flat_to_group, "block_ids": block_ids,
        "roles": role_labels, "energy": energy_flat,
    }

    descriptor_fn = make_descriptor_fn(flat_to_group, level_costs_flat, energy_flat, attn_flat, block_ids)
    bounds_fn = make_bounds_fn()

    # CVT samples (biased)
    print(f"Sampling {args.cvt_samples} feasible genomes for CVT...")
    biases = ["uniform", "attn_heavy", "mlp_heavy"]
    samples = []
    for i in trange(args.cvt_samples, desc="CVT sampling", leave=False):
        samples.append(sample_random_genome(grouped_names, level_costs, n_levels, budget,
                                             flat_to_group, attn_flat=attn_flat,
                                             bias=biases[i % len(biases)]))

    basins = derive_basins_from_samples(samples=samples, descriptor_fn=descriptor_fn,
                                        bounds_fn=bounds_fn, k=args.num_islands,
                                        iters=args.cvt_iters, name_prefix="lr", seed=args.seed)
    print(f"CVT produced {len(basins)} basins:")
    for b in basins:
        print(f"  {b.name} mass={b.descriptor_mass} centroid={tuple(round(c,2) for c in b.descriptor_centroid)} r={b.descriptor_radius:.2f}")

    # Fitness-based island seeding
    print(f"Evaluating {len(samples)} CVT samples for fitness-based seeding...")
    init_mb, init_lmb = sample_minibatch(calib_data, args.tokens_per_selection[0],
                                         target_logits if args.fitness_fn == "kl" else None)
    sample_fits = []
    for g in samples:
        load_layers(model, grouped_names, g, args.lowrank_db)
        sample_fits.append(compute_fitness(model, init_mb, args.fitness_fn, init_lmb))
    order = sorted(range(len(samples)), key=lambda i: sample_fits[i])
    fit_rank = [0.0] * len(samples)
    for rk, i in enumerate(order):
        fit_rank[i] = rk / max(1, len(samples) - 1)
    print(f"  CVT init best fit: {sample_fits[order[0]]:.4e} worst: {sample_fits[order[-1]]:.4e}")

    islands = []
    island_prefs = []
    alpha = 0.5
    for basin in basins:
        bandit = bandit_for_basin(basin)
        lpref = basin_layer_pref(basin, attn_flat, mlp_flat, block_ids)
        def score(i, basin=basin):
            g = samples[i]
            d = math.sqrt(sum(((dd - c) / max(s, 1e-6)) ** 2
                              for dd, c, s in zip(descriptor_fn(g), basin.descriptor_centroid, basin.descriptor_scale)))
            return d + alpha * fit_rank[i]
        chosen = [samples[i] for i in sorted(range(len(samples)), key=score)[:args.pop_size]]
        pop = [clone_genome(g) for g in chosen]
        while len(pop) < args.pop_size:
            pop.append(sample_random_genome(grouped_names, level_costs, n_levels, budget,
                                            flat_to_group, attn_flat=attn_flat, bias="uniform"))
        islands.append(Island(name=basin.name, basin=basin, pop=pop, bandit=bandit,
                              stag_threshold=args.stag_threshold))
        island_prefs.append(lpref)

    hof = HallOfFame(capacity=args.hof_capacity, sig_fn=genome_sig)
    global_best_fitness = float("inf")
    global_best_genome = clone_genome(islands[0].pop[0])
    log_dict = {}

    print(f"\nRunning {args.generations} gens × {len(islands)} islands.")
    for gen in trange(args.generations, desc="Generations"):
        for island, lpref in zip(islands, island_prefs):
            new_pop, best_f, op_counts = island_generation(
                island, lpref, descriptor_fn, ctx, model, grouped_names, args.lowrank_db,
                calib_data, target_logits if target_logits else None,
                args.offspring, args.pop_size, args.survivors_per_selection,
                args.tokens_per_selection, args.fitness_fn, hof, args.crossover_rate)
            island.pop = new_pop
            island.record(best_f)
            print(f"  [gen {gen}] {island.name} best_f={best_f:.4e} stag={island.stag}")
            if best_f < global_best_fitness:
                global_best_fitness = best_f
                global_best_genome = clone_genome(new_pop[0])

        # Hill-climb every 5 gens
        if (gen + 1) % 5 == 0:
            hc_tok = args.tokens_per_selection[-1]
            hc_mb, hc_lmb = sample_minibatch(calib_data, hc_tok, target_logits if args.fitness_fn == "kl" else None)
            def _hc_eval(cg):
                load_layers(model, grouped_names, cg, args.lowrank_db)
                if args.fitness_fn == "ppl":
                    return compute_perplexity(model, hc_mb)
                return compute_kl_div(model, hc_mb, hc_lmb)
            baseline_f = _hc_eval(global_best_genome)
            hc_g, hc_f = hill_climb_elite(global_best_genome, ctx, _hc_eval)
            if hc_f < baseline_f:
                print(f"  [gen {gen}] HILL-CLIMB {baseline_f:.4e} → {hc_f:.4e}")
                global_best_fitness = hc_f
                global_best_genome = hc_g
                hof.try_add(hc_g, hc_f)
                islands[0].pop[0] = clone_genome(hc_g)

        if (gen + 1) % args.migration_every == 0:
            bests = [clone_genome(isl.pop[0]) for isl in islands]
            for i, isl in enumerate(islands):
                others = [bests[j] for j in range(len(islands)) if j != i]
                random.shuffle(others)
                present = {genome_sig(p) for p in isl.pop}
                for k in range(min(2, len(isl.pop) - 1, len(others))):
                    if genome_sig(others[k]) not in present:
                        isl.pop[-(k + 1)] = others[k]
                        present.add(genome_sig(others[k]))

        for island, lpref in zip(islands, island_prefs):
            if island.is_stagnant():
                warm_restart_island(island, hof, ctx, lpref, attn_flat)
        if args.restart_every > 0 and (gen + 1) % args.restart_every == 0:
            for island, lpref in zip(islands, island_prefs):
                warm_restart_island(island, hof, ctx, lpref, attn_flat)

        if gen % args.eval_every == 0:
            load_layers(model, grouped_names, global_best_genome, args.lowrank_db)
            for dname, dset in zip(args.eval_datasets, eval_datasets):
                ppl = compute_perplexity(model, dset)
                print(f"  [{dname}] ppl={ppl:.3f}")
                log_dict[f"ppl_eval/{dname}"] = ppl
            cost = _cost_of(global_best_genome, grouped_names, level_costs, flat_to_group)
            print(f"  fit={global_best_fitness:.3e}  ratio={cost/total_orig:.4f}")
            for island in islands:
                print(f"  [{island.name}] {island.bandit.summary()}")
        if args.log_wandb:
            wandb.log(log_dict)

    hof_g, hof_f = hof.best()
    if hof_g is not None and hof_f < global_best_fitness:
        global_best_genome, global_best_fitness = hof_g, hof_f

    out_path = os.path.join(args.lowrank_db, args.configuration_name)
    with open(out_path, "w") as f:
        lines = []
        for gi, group in enumerate(grouped_names):
            for name, lvl in zip(group, global_best_genome[gi]):
                lines.append(f"{name}: {lvl}")
        f.write("\n".join(lines))
    print(f"\nSaved → {out_path}")

    load_layers(model, grouped_names, global_best_genome, args.lowrank_db)
    for dname, dset in zip(args.eval_datasets, eval_datasets):
        print(f"FINAL {dname}: {compute_perplexity(model, dset):.3f}")


if __name__ == "__main__":
    main()
