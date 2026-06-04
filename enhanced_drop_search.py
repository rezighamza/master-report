"""
Enhanced layer-drop (depth-pruning) search for EvoPress (CVT island model).

Genome:       {"attn": [bool]*L, "mlp": [bool]*L}   (True = dropped)
Constraint:   exact number of drops (target_sparsity * 2L sub-blocks)
Descriptors:  6D — (drop_com, block_drop_share, attn_only_share,
                    mlp_only_share, drop_clustering, total_drops_norm)
"""
from __future__ import annotations

import argparse
import copy
import math
import os
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
from src.model_utils import (
    get_layers, get_attn_layer_name, get_mlp_layer_name,
    make_dummy_forward, dummy_initialize, restore_forward,
)
from src.search_utils import (
    EXP3Bandit, HallOfFame, CosineAnnealingTemperature, BasinSpec, Island,
    derive_basins_from_samples, in_basin, sample_minibatch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Genome utilities
# ─────────────────────────────────────────────────────────────────────────────

def genome_sig(g: Dict[str, List[bool]]) -> Tuple:
    return (tuple(g["attn"]), tuple(g["mlp"]))


def clone_genome(g: Dict[str, List[bool]]) -> Dict[str, List[bool]]:
    return {"attn": list(g["attn"]), "mlp": list(g["mlp"])}


def total_drops(g: Dict[str, List[bool]]) -> int:
    return sum(g["attn"]) + sum(g["mlp"])


def apply_state(model, layers, g: Dict[str, List[bool]]):
    """Apply drop state via make_dummy_forward / restore_forward."""
    attn_name = get_attn_layer_name(model)
    mlp_name  = get_mlp_layer_name(model)
    for j in range(len(g["attn"])):
        sub = getattr(layers[j], attn_name)
        if g["attn"][j]:
            make_dummy_forward(sub, "attn")
        else:
            restore_forward(sub)
    for j in range(len(g["mlp"])):
        sub = getattr(layers[j], mlp_name)
        if g["mlp"][j]:
            make_dummy_forward(sub, "mlp")
        else:
            restore_forward(sub)


# ─────────────────────────────────────────────────────────────────────────────
# Descriptors (6D)
# ─────────────────────────────────────────────────────────────────────────────

def make_descriptor_fn(L: int):
    """
    drop_com           : centre-of-mass of drop positions in [0,1] (0=early, 1=late)
    block_drop_share   : fraction of drops that are full-block (attn+mlp)
    attn_only_share    : fraction of drops that are attn-only
    mlp_only_share     : fraction of drops that are mlp-only
    drop_clustering    : 1 - normalised stddev of inter-drop gaps (1=clustered, 0=spread)
    total_drops_norm   : count(dropped) / (2*L)
    """
    def desc(g: Dict[str, List[bool]]) -> Tuple[float, ...]:
        a, m = g["attn"], g["mlp"]
        positions = []
        block_n = attn_n = mlp_n = 0
        for i in range(L):
            both = a[i] and m[i]
            ao = a[i] and not m[i]
            mo = m[i] and not a[i]
            if both:
                positions.append(i); block_n += 1
            if ao:
                positions.append(i); attn_n += 1
            if mo:
                positions.append(i); mlp_n += 1
        total = block_n + attn_n + mlp_n
        if total == 0:
            return (0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
        com = (sum(positions) / total) / max(1, L - 1)
        block_s = block_n / total
        attn_s  = attn_n / total
        mlp_s   = mlp_n / total
        if len(positions) >= 2:
            ps = sorted(positions)
            gaps = [ps[i+1] - ps[i] for i in range(len(ps) - 1)]
            std_norm = (np.std(gaps) / max(np.mean(gaps), 1e-6)) if gaps else 0.0
            clust = max(0.0, 1.0 - min(1.0, std_norm))
        else:
            clust = 1.0
        total_norm = total / max(1, 2 * L)
        return (com, block_s, attn_s, mlp_s, clust, total_norm)
    return desc


def make_bounds_fn():
    def bounds(members: List[Dict[str, List[bool]]]) -> Dict[str, Any]:
        drops = [total_drops(g) for g in members]
        return {
            "drops_lo": int(np.quantile(drops, 0.05)) if drops else 0,
            "drops_hi": int(np.quantile(drops, 0.95)) if drops else 0,
        }
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# Random feasible sampler (CVT seed pool)
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_genome(L: int, target_drops: int, legal_mask, bias: str = "uniform") -> Dict[str, List[bool]]:
    g = {"attn": [False] * L, "mlp": [False] * L}
    positions: List[Tuple[str, int]] = []
    if bias == "block":
        # Prefer dropping whole blocks
        blocks = list(range(L)); random.shuffle(blocks)
        for b in blocks:
            if legal_mask["attn"][b] and legal_mask["mlp"][b]:
                positions.append(("attn", b)); positions.append(("mlp", b))
                if len(positions) >= target_drops:
                    break
    elif bias == "attn_only":
        idx = [i for i in range(L) if legal_mask["attn"][i]]
        random.shuffle(idx)
        for i in idx[:target_drops]:
            positions.append(("attn", i))
    elif bias == "mlp_only":
        idx = [i for i in range(L) if legal_mask["mlp"][i]]
        random.shuffle(idx)
        for i in idx[:target_drops]:
            positions.append(("mlp", i))
    elif bias in ("early", "late"):
        half = L // 2
        zone = range(half) if bias == "early" else range(half, L)
        pool = []
        for i in zone:
            if legal_mask["attn"][i]: pool.append(("attn", i))
            if legal_mask["mlp"][i]:  pool.append(("mlp", i))
        random.shuffle(pool)
        positions = pool[:target_drops]
        if len(positions) < target_drops:
            extra = []
            for i in range(L):
                if i in zone: continue
                if legal_mask["attn"][i]: extra.append(("attn", i))
                if legal_mask["mlp"][i]:  extra.append(("mlp", i))
            random.shuffle(extra)
            positions += extra[: target_drops - len(positions)]
    else:  # uniform
        pool = []
        for i in range(L):
            if legal_mask["attn"][i]: pool.append(("attn", i))
            if legal_mask["mlp"][i]:  pool.append(("mlp", i))
        random.shuffle(pool)
        positions = pool[:target_drops]

    for sub, i in positions:
        g[sub][i] = True
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Mutation operators (preserve exact drop count)
# ─────────────────────────────────────────────────────────────────────────────

def _dropped_positions(g: Dict[str, List[bool]]) -> List[Tuple[str, int]]:
    return [("attn", i) for i, v in enumerate(g["attn"]) if v] + \
           [("mlp", i) for i, v in enumerate(g["mlp"]) if v]

def _kept_positions(g: Dict[str, List[bool]], legal_mask) -> List[Tuple[str, int]]:
    L = len(g["attn"])
    out = []
    for i in range(L):
        if not g["attn"][i] and legal_mask["attn"][i]: out.append(("attn", i))
        if not g["mlp"][i]  and legal_mask["mlp"][i]:  out.append(("mlp", i))
    return out


def op_swap_one(g, legal_mask, layer_pref=None):
    """Move one drop from a dropped position to a kept position."""
    g2 = clone_genome(g)
    dropped = _dropped_positions(g2)
    kept = _kept_positions(g2, legal_mask)
    if not dropped or not kept:
        return g2
    # Bias destination by layer_pref if given
    if layer_pref:
        kept_pref = [(s, i) for (s, i) in kept if i in layer_pref]
        dst = random.choice(kept_pref) if kept_pref and random.random() < 0.8 else random.choice(kept)
    else:
        dst = random.choice(kept)
    src = random.choice(dropped)
    g2[src[0]][src[1]] = False
    g2[dst[0]][dst[1]] = True
    return g2


def op_swap_two(g, legal_mask, layer_pref=None):
    g2 = op_swap_one(g, legal_mask, layer_pref)
    return op_swap_one(g2, legal_mask, layer_pref)


def op_swap_three(g, legal_mask, layer_pref=None):
    g2 = op_swap_one(g, legal_mask, layer_pref)
    g2 = op_swap_one(g2, legal_mask, layer_pref)
    return op_swap_one(g2, legal_mask, layer_pref)


def op_convert_to_block(g, legal_mask, layer_pref=None):
    """Convert an attn-only drop at layer i into a (attn+mlp) drop, freeing a kept attn/mlp elsewhere."""
    g2 = clone_genome(g)
    L = len(g2["attn"])
    # find an attn-only or mlp-only drop
    candidates = [i for i in range(L) if (g2["attn"][i] != g2["mlp"][i])]
    if not candidates:
        return op_swap_one(g, legal_mask, layer_pref)
    i = random.choice(candidates)
    # find a kept full block to use as source of one drop
    kept_full = [j for j in range(L) if (not g2["attn"][j] and not g2["mlp"][j]
                                          and legal_mask["attn"][j] and legal_mask["mlp"][j])]
    dropped = _dropped_positions(g2)
    if not dropped:
        return op_swap_one(g, legal_mask, layer_pref)
    # complete the partial block (drop the kept side too) and remove a drop elsewhere
    if not g2["attn"][i] and legal_mask["attn"][i]:
        g2["attn"][i] = True
    elif not g2["mlp"][i] and legal_mask["mlp"][i]:
        g2["mlp"][i] = True
    src = random.choice([d for d in dropped if not (d[1] == i)])
    g2[src[0]][src[1]] = False
    return g2


def op_split_block(g, legal_mask, layer_pref=None):
    """Inverse of convert_to_block: split a full-block drop into attn-only + mlp-only spread."""
    g2 = clone_genome(g)
    L = len(g2["attn"])
    full_blocks = [i for i in range(L) if g2["attn"][i] and g2["mlp"][i]]
    if not full_blocks:
        return op_swap_one(g, legal_mask, layer_pref)
    i = random.choice(full_blocks)
    # remove the mlp drop and put it elsewhere
    g2["mlp"][i] = False
    kept_mlp = [j for j in range(L) if not g2["mlp"][j] and j != i and legal_mask["mlp"][j]]
    if kept_mlp:
        j = random.choice(kept_mlp)
        g2["mlp"][j] = True
    else:
        g2["mlp"][i] = True   # revert if no space
    return g2


def op_crossover(g1, g2, legal_mask, layer_pref=None):
    """Per-layer crossover, then repair to exact target count."""
    L = len(g1["attn"])
    target = total_drops(g1)
    child = {"attn": [], "mlp": []}
    for i in range(L):
        if random.random() < 0.5:
            child["attn"].append(g1["attn"][i]); child["mlp"].append(g1["mlp"][i])
        else:
            child["attn"].append(g2["attn"][i]); child["mlp"].append(g2["mlp"][i])
    # Repair: add/remove drops to reach target
    delta = total_drops(child) - target
    while delta > 0:   # too many drops, remove one
        dropped = _dropped_positions(child)
        s, i = random.choice(dropped)
        child[s][i] = False
        delta -= 1
    while delta < 0:   # too few drops, add one
        kept = _kept_positions(child, legal_mask)
        if not kept: break
        s, i = random.choice(kept)
        child[s][i] = True
        delta += 1
    return child


OPERATORS = ["swap_one", "swap_two", "swap_three", "convert_to_block", "split_block"]
OP_FNS = {
    "swap_one": op_swap_one,
    "swap_two": op_swap_two,
    "swap_three": op_swap_three,
    "convert_to_block": op_convert_to_block,
    "split_block": op_split_block,
}


# ─────────────────────────────────────────────────────────────────────────────
# Basin → bandit priors and layer preferences
# ─────────────────────────────────────────────────────────────────────────────

def bandit_for_basin(basin: BasinSpec, gamma: float = 0.07) -> EXP3Bandit:
    """
    Bias by basin's (block_share, attn_only_share, mlp_only_share, clustering).
    """
    b = EXP3Bandit(OPERATORS, gamma=gamma)
    _, block_s, attn_s, mlp_s, clust, _ = basin.descriptor_centroid
    if block_s > max(attn_s, mlp_s):
        b.w.update({"convert_to_block": 3.0, "swap_one": 2.0, "swap_two": 1.0,
                    "split_block": 0.5, "swap_three": 0.5})
    elif attn_s + mlp_s > 2 * block_s:
        b.w.update({"split_block": 3.0, "swap_one": 2.0, "swap_two": 1.5,
                    "convert_to_block": 0.5, "swap_three": 0.5})
    else:
        b.w.update({"swap_one": 2.5, "swap_two": 2.0, "swap_three": 1.5,
                    "convert_to_block": 1.0, "split_block": 1.0})
    return b


def basin_layer_pref(basin: BasinSpec, L: int) -> Optional[List[int]]:
    """drop_com near 0 → prefer early layers; near 1 → prefer late layers."""
    com = basin.descriptor_centroid[0]
    half = L // 2
    if com <= 0.35:
        return list(range(0, half))
    if com >= 0.65:
        return list(range(half, L))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────────────────────

def compute_fitness(model, data, fitness_fn, target_logits=None) -> float:
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    return compute_kl_div(model, data, target_logits)


def eval_pool(model, layers, pool, calib_data, num_tokens, fitness_fn, target_logits):
    mb, lmb = sample_minibatch(calib_data, num_tokens, target_logits if fitness_fn == "kl" else None)
    out = []
    for g in pool:
        apply_state(model, layers, g)
        out.append(compute_fitness(model, mb, fitness_fn, lmb))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Island generation
# ─────────────────────────────────────────────────────────────────────────────

def island_generation(
    island: Island, layer_pref, descriptor_fn, legal_mask,
    model, layers, calib_data, target_logits,
    n_offspring: int, pop_size: int,
    survivors_per_selection, tokens_per_selection,
    fitness_fn: str, hof: HallOfFame, crossover_rate: float,
):
    bandit = island.bandit
    basin = island.basin
    population = island.pop

    offspring = []
    op_log: List[Tuple[str, int]] = []
    seen = {genome_sig(p) for p in population}
    op_counts = {op: 0 for op in OPERATORS + ["crossover"]}

    attempts = 0
    while len(offspring) < n_offspring and attempts < n_offspring * 30:
        attempts += 1
        if len(population) >= 2 and random.random() < crossover_rate:
            p1, p2 = random.sample(population, 2)
            child = op_crossover(p1, p2, legal_mask, layer_pref)
            op = bandit.sample()
            child = OP_FNS[op](child, legal_mask, layer_pref)
            op_counts["crossover"] += 1
        else:
            parent = random.choice(population)
            op = bandit.sample()
            child = OP_FNS[op](parent, legal_mask, layer_pref)
        # Basin repair
        if not in_basin(descriptor_fn(child), basin):
            op2 = bandit.sample()
            cand = OP_FNS[op2](child, legal_mask, layer_pref)
            if in_basin(descriptor_fn(cand), basin):
                child = cand; op = op2
        sig = genome_sig(child)
        if sig in seen:
            continue
        seen.add(sig)
        op_log.append((op, len(offspring)))
        offspring.append(child)
        op_counts[op] += 1

    elite_pool = [p for p in population if genome_sig(p) not in {genome_sig(c) for c in offspring}]
    full_pool = offspring + elite_pool

    stage_survivors = list(survivors_per_selection[:-1]) + [pop_size]
    pool = list(full_pool)
    final_fitnesses = [float("inf")] * len(pool)
    for n_surv, n_tok in zip(stage_survivors, tokens_per_selection):
        n_surv = min(n_surv, len(pool))
        fits = eval_pool(model, layers, pool, calib_data, n_tok, fitness_fn, target_logits)
        best_ids = list(np.argsort(fits)[:n_surv])
        pool = [pool[i] for i in best_ids]
        final_fitnesses = [fits[i] for i in best_ids]

    for g, f in zip(pool, final_fitnesses):
        hof.try_add(g, f)

    survivor_sigs = {genome_sig(s) for s in pool}
    for op, cidx in op_log:
        if cidx < len(offspring):
            reward = 1.0 if genome_sig(offspring[cidx]) in survivor_sigs else 0.0
            bandit.update(op, reward)

    return pool, (final_fitnesses[0] if final_fitnesses else float("inf")), op_counts


def warm_restart_island(island: Island, hof: HallOfFame, legal_mask, layer_pref,
                         target_drops: int, L: int):
    """
    Diversifying restart:
      slot 0       : keep current best (elite preservation)
      next ~1/3    : HoF seeds (exploit good basins)
      remaining    : FRESH random feasible genomes with varied biases (true exploration)
    Also resets the bandit weights so it re-explores all operators.
    """
    pop_size = len(island.pop)
    if pop_size == 0:
        return
    new_pop = [clone_genome(island.pop[0])]  # keep current island best

    n_hof = max(0, pop_size // 3)
    if n_hof > 0 and len(hof) > 0:
        for seed in hof.sample_seeds(n_hof):
            new_pop.append(clone_genome(seed))

    biases = ["uniform", "block", "attn_only", "mlp_only", "early", "late"]
    while len(new_pop) < pop_size:
        bias = random.choice(biases)
        new_pop.append(sample_random_genome(L, target_drops, legal_mask, bias=bias))

    island.pop = new_pop
    # Reset bandit weights so locked-in operators get re-explored
    for op in island.bandit.ops:
        island.bandit.w[op] = 1.0
    island.stag = 0
    island.best_fitness = float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Args + Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True, type=str)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--calibration_data", required=True, type=str)
    p.add_argument("--calibration_tokens", required=True, type=int)
    p.add_argument("--calibration_sequence_length", default=None, type=int)
    p.add_argument("--eval_datasets", nargs="+", type=str, default=["fineweb_edu", "wikitext2", "c4"])
    p.add_argument("--eval_every", default=1, type=int)
    p.add_argument("--eval_tokens", default=524288, type=int)
    p.add_argument("--eval_sequence_length", default=None, type=int)
    p.add_argument("--fitness_fn", choices=["ppl", "kl"], default="kl")
    p.add_argument("--sparsity", required=True, type=float,
                   help="Target drop fraction of sub-blocks (e.g. 0.25 = drop 25% of attn/mlp)")
    p.add_argument("--drop_entire_block", action="store_true",
                   help="Initialise drops as full blocks (attn+mlp paired)")
    p.add_argument("--legal_to_drop_path", type=str, default=None)
    p.add_argument("--generations", required=True, type=int)
    p.add_argument("--offspring", required=True, type=int)
    p.add_argument("--survivors_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--tokens_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--num_islands", type=int, default=4)
    p.add_argument("--cvt_samples", type=int, default=300)
    p.add_argument("--cvt_iters", type=int, default=30)
    p.add_argument("--pop_size", type=int, default=4)
    p.add_argument("--migration_every", type=int, default=5)
    p.add_argument("--restart_every", type=int, default=50)
    p.add_argument("--stag_threshold", type=int, default=10)
    p.add_argument("--crossover_rate", type=float, default=0.40)
    p.add_argument("--hof_capacity", type=int, default=30)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--attn_implementation", default=None,
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--memory_efficient", action="store_true")
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--log_wandb", default=False, action="store_true")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--configuration_name", type=str, default="layer_drop_config.txt")
    return p.parse_args()


def get_legal_mask(path: Optional[str], L: int):
    if path is None:
        return {"attn": [True] * L, "mlp": [True] * L}
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines()]
    assert len(lines) == L
    mask = {"attn": [False] * L, "mlp": [False] * L}
    for i, t in enumerate(lines):
        if t == "attn+mlp":
            mask["attn"][i] = True; mask["mlp"][i] = True
        elif t == "attn":
            mask["attn"][i] = True
        elif t == "mlp":
            mask["mlp"][i] = True
    return mask


def get_drop_config(g: Dict[str, List[bool]]) -> List[str]:
    L = len(g["attn"])
    out = ["none"] * L
    for i in range(L):
        if g["attn"][i] and g["mlp"][i]:
            out[i] = "attn+mlp"
        elif g["attn"][i]:
            out[i] = "attn"
        elif g["mlp"][i]:
            out[i] = "mlp"
    return out


def main():
    args = parse_args()
    fix_seed(args.seed)
    if args.log_wandb:
        assert has_wandb
        wandb.init(config=args)

    device = "cuda"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None if args.memory_efficient else "auto",
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )

    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings, 8192)
    calib_data = get_data(args.calibration_data, args.calibration_tokens,
                          args.calibration_sequence_length, tokenizer, train=True)
    args.eval_sequence_length = args.eval_sequence_length or min(
        model.config.max_position_embeddings, 8192)
    eval_datasets = [get_data(n, args.eval_tokens, args.eval_sequence_length, tokenizer, train=False)
                     for n in args.eval_datasets]

    target_logits = []
    if args.fitness_fn == "kl":
        for i in trange(len(calib_data), desc="Target logits", leave=False):
            with torch.no_grad():
                target_logits.append(model(calib_data[i].to(device)).logits.cpu())

    layers = get_layers(model)
    L = len(layers)
    # Initialise dummy_forward so make/restore work
    attn_name = get_attn_layer_name(model)
    mlp_name  = get_mlp_layer_name(model)
    for j in range(L):
        dummy_initialize(getattr(layers[j], attn_name))
        dummy_initialize(getattr(layers[j], mlp_name))

    legal_mask = get_legal_mask(args.legal_to_drop_path, L)
    target_drops = int(round(args.sparsity * 2 * L))
    print(f"Layers={L}  target_drops={target_drops} (sparsity={args.sparsity})")

    descriptor_fn = make_descriptor_fn(L)
    bounds_fn = make_bounds_fn()

    print(f"Sampling {args.cvt_samples} feasible genomes for CVT...")
    biases = ["uniform", "block", "attn_only", "mlp_only", "early", "late"]
    samples = []
    for i in trange(args.cvt_samples, desc="CVT sampling", leave=False):
        bias = biases[i % len(biases)] if not args.drop_entire_block else "block"
        samples.append(sample_random_genome(L, target_drops, legal_mask, bias=bias))

    basins = derive_basins_from_samples(
        samples=samples, descriptor_fn=descriptor_fn, bounds_fn=bounds_fn,
        k=args.num_islands, iters=args.cvt_iters, name_prefix="drop", seed=args.seed,
    )
    print(f"CVT produced {len(basins)} basins:")
    for b in basins:
        print(f"  {b.name}  mass={b.descriptor_mass}  centroid={tuple(round(c,2) for c in b.descriptor_centroid)}  r={b.descriptor_radius:.2f}")

    islands: List[Island] = []
    island_prefs: List[Optional[List[int]]] = []
    for basin in basins:
        bandit = bandit_for_basin(basin)
        lpref = basin_layer_pref(basin, L)
        nearest = sorted(samples, key=lambda g: math.sqrt(sum(
            ((d - c) / max(s, 1e-6)) ** 2
            for d, c, s in zip(descriptor_fn(g), basin.descriptor_centroid, basin.descriptor_scale)
        )))[:args.pop_size]
        pop = [clone_genome(g) for g in nearest]
        while len(pop) < args.pop_size:
            pop.append(sample_random_genome(L, target_drops, legal_mask, bias="uniform"))
        islands.append(Island(name=basin.name, basin=basin, pop=pop, bandit=bandit,
                               stag_threshold=args.stag_threshold))
        island_prefs.append(lpref)

    hof = HallOfFame(capacity=args.hof_capacity, sig_fn=genome_sig)
    temperature = CosineAnnealingTemperature(T_max=3.0, T_min=1.0,
                                              period=max(10, args.generations // 3))

    global_best_fitness = float("inf")
    global_best_genome = clone_genome(islands[0].pop[0])
    log_dict: Dict[str, Any] = {}

    print(f"\nRunning {args.generations} gens × {len(islands)} islands.")
    for gen in trange(args.generations, desc="Generations"):
        T = temperature.step()
        for island, lpref in zip(islands, island_prefs):
            new_pop, best_f, op_counts = island_generation(
                island=island, layer_pref=lpref, descriptor_fn=descriptor_fn, legal_mask=legal_mask,
                model=model, layers=layers, calib_data=calib_data,
                target_logits=target_logits if target_logits else None,
                n_offspring=args.offspring, pop_size=args.pop_size,
                survivors_per_selection=args.survivors_per_selection,
                tokens_per_selection=args.tokens_per_selection,
                fitness_fn=args.fitness_fn, hof=hof, crossover_rate=args.crossover_rate,
            )
            island.pop = new_pop
            island.record(best_f)
            log_dict[f"train_fitness/{island.name}"] = best_f
            log_dict[f"stag/{island.name}"] = island.stag
            if best_f < global_best_fitness:
                global_best_fitness = best_f
                global_best_genome = clone_genome(new_pop[0])

        log_dict["train_fitness/global_best"] = global_best_fitness
        log_dict["temperature"] = T

        if (gen + 1) % args.migration_every == 0:
            bests = [clone_genome(isl.pop[0]) for isl in islands]
            for i, isl in enumerate(islands):
                others = [bests[j] for j in range(len(islands)) if j != i]
                random.shuffle(others)
                present = {genome_sig(p) for p in isl.pop}
                n_replace = min(2, len(isl.pop) - 1, len(others))
                for k in range(n_replace):
                    donor = others[k]
                    if genome_sig(donor) not in present:
                        isl.pop[-(k + 1)] = donor
                        present.add(genome_sig(donor))

        for island, lpref in zip(islands, island_prefs):
            if island.is_stagnant():
                warm_restart_island(island, hof, legal_mask, lpref, target_drops, L)

        if args.restart_every > 0 and (gen + 1) % args.restart_every == 0:
            for island, lpref in zip(islands, island_prefs):
                warm_restart_island(island, hof, legal_mask, lpref, target_drops, L)

        if gen % args.eval_every == 0:
            apply_state(model, layers, global_best_genome)
            for dname, dset in zip(args.eval_datasets, eval_datasets):
                ppl = compute_perplexity(model, dset)
                print(f"  [{dname}] ppl={ppl:.3f}")
                log_dict[f"ppl_eval/{dname}"] = ppl
            print(f"  fit={global_best_fitness:.3e}  T={T:.2f}")
            log_dict["gen"] = gen
            for island in islands:
                print(f"  [{island.name}] {island.bandit.summary()}")
        if args.log_wandb:
            wandb.log(log_dict)

    # Final
    hof_g, hof_f = hof.best()
    if hof_g is not None and hof_f < global_best_fitness:
        global_best_genome = hof_g
        global_best_fitness = hof_f
    apply_state(model, layers, global_best_genome)

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.configuration_name)
    with open(out_path, "w") as f:
        f.write("\n".join(get_drop_config(global_best_genome)))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
