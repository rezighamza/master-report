"""
Enhanced unstructured-sparsity search for EvoPress (CVT island model).

Genome:        List[int]                  — per-layer sparsity level (0 = baseline)
Constraint:    sum(|v_i|) <= max_total_deviation
Descriptors:   6D — (attn_share, mlp_share, early_share, late_share,
                     layer_entropy, positive_share)

Islands are CVT cells over the descriptor space, derived by k-means on 300
random valid genomes sampled before the search starts. Each island has an
EXP3-IX bandit, a BasinSpec, and stagnation-driven warm restart from HoF.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from tqdm import trange
from typing import Dict, List, Optional, Tuple, Any

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
from src.search_utils import (
    EXP3Bandit, HallOfFame, CosineAnnealingTemperature,
    BasinSpec, Island, derive_basins_from_samples, in_basin, sample_minibatch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading
# ─────────────────────────────────────────────────────────────────────────────

def load_layers(model, layer_names: List[str], new_state: List[int], weights_path: str):
    assert hasattr(model, "state")
    for name, new_lv, old_lv in zip(layer_names, new_state, model.state):
        if new_lv != old_lv:
            layer = model.get_submodule(name)
            layer.weight.data = torch.load(
                os.path.join(weights_path, name, f"{new_lv}.pth"),
                map_location=layer.weight.device,
            ).to(layer.weight.dtype)
    model.state = list(new_state)


# ─────────────────────────────────────────────────────────────────────────────
# Genome helpers
# ─────────────────────────────────────────────────────────────────────────────

def genome_cost(g: List[int]) -> int:
    return sum(abs(v) for v in g)

def _file_ok(weights_path: str, name: str, level: int) -> bool:
    return os.path.exists(os.path.join(weights_path, name, f"{level}.pth"))


def classify_layers(layer_names: List[str]) -> Dict[str, List[int]]:
    """Partition layer indices by sub-module type and depth."""
    attn, mlp = [], []
    for i, name in enumerate(layer_names):
        if "self_attn" in name:
            attn.append(i)
        elif "mlp" in name:
            mlp.append(i)
    L = len(layer_names)
    half = L // 2
    return {
        "attn": attn,
        "mlp": mlp,
        "early": list(range(half)),
        "late": list(range(half, L)),
        "all": list(range(L)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Descriptors (6D, normalised to [0,1])
# ─────────────────────────────────────────────────────────────────────────────

def make_descriptor_fn(layer_groups: Dict[str, List[int]]):
    """
    6D descriptor:
      attn_share       : fraction of |v| on attention projections
      mlp_share        : fraction of |v| on MLP projections
      early_share      : fraction of |v| on first half of layers
      late_share       : fraction of |v| on second half of layers
      layer_entropy    : normalised entropy of |v| across layers (0=concentrated, 1=spread)
      positive_share   : fraction of layers with v_i > 0 (sparser-than-baseline)
    """
    attn = layer_groups["attn"]
    mlp = layer_groups["mlp"]
    early = layer_groups["early"]
    late = layer_groups["late"]
    L = len(layer_groups["all"])
    logL = math.log(max(2, L))

    def desc(g: List[int]) -> Tuple[float, ...]:
        abs_g = [abs(v) for v in g]
        total = sum(abs_g)
        if total <= 0:
            return (0.0, 0.0, 0.5, 0.5, 1.0, 0.0)
        attn_s = sum(abs_g[i] for i in attn) / total
        mlp_s  = sum(abs_g[i] for i in mlp)  / total
        early_s = sum(abs_g[i] for i in early) / total
        late_s  = sum(abs_g[i] for i in late)  / total
        p = [a / total for a in abs_g if a > 0]
        ent = -sum(pi * math.log(pi) for pi in p) / logL if p else 0.0
        pos_s = sum(1 for v in g if v > 0) / L
        return (attn_s, mlp_s, early_s, late_s, ent, pos_s)
    return desc


def make_bounds_fn():
    """Per-cell scalar bounds: max-level cap derived from member quantiles."""
    def bounds(members: List[List[int]]) -> Dict[str, Any]:
        max_abs = max((max(abs(v) for v in g) for g in members), default=1)
        budgets = [sum(abs(v) for v in g) for g in members]
        return {
            "max_abs_level":   int(max_abs),
            "budget_lo":       int(np.quantile(budgets, 0.05)) if budgets else 0,
            "budget_hi":       int(np.quantile(budgets, 0.95)) if budgets else 0,
        }
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# Random feasible genome sampler (for CVT seed pool)
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_genome(
    layer_names: List[str],
    weights_path: str,
    max_level: int,
    max_dev: int,
    available_levels: List[List[int]],
    bias: str = "uniform",
    layer_groups: Optional[Dict[str, List[int]]] = None,
) -> List[int]:
    """
    Sample a random valid genome under the budget. `bias` shapes WHERE the
    budget is spent so CVT sees a diverse population.
    """
    n = len(layer_names)
    g = [0] * n
    target = random.randint(max(1, max_dev // 4), max_dev)
    spent = 0
    pool = list(range(n))
    if bias == "attn" and layer_groups:
        pool = layer_groups["attn"] * 3 + layer_groups["mlp"]
    elif bias == "mlp" and layer_groups:
        pool = layer_groups["mlp"] * 3 + layer_groups["attn"]
    elif bias == "early" and layer_groups:
        pool = layer_groups["early"] * 3 + layer_groups["late"]
    elif bias == "late" and layer_groups:
        pool = layer_groups["late"] * 3 + layer_groups["early"]

    attempts = 0
    while spent < target and attempts < target * 20:
        attempts += 1
        i = random.choice(pool)
        direction = 1 if random.random() < 0.85 else -1   # mostly increase sparsity
        new_lv = g[i] + direction
        if abs(new_lv) > max_level:
            continue
        if not _file_ok(weights_path, layer_names[i], new_lv):
            continue
        if spent + (abs(new_lv) - abs(g[i])) > max_dev:
            continue
        spent += abs(new_lv) - abs(g[i])
        g[i] = new_lv
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Mutation operators (basin-aware via layer_pref)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_layer(n: int, layer_pref: Optional[List[int]], affinity: float = 0.8) -> int:
    if layer_pref and random.random() < affinity:
        return random.choice(layer_pref)
    return random.randrange(n)


def _flip_pair(g, names, path, max_level, max_dev, layer_pref=None):
    n = len(g)
    for _ in range(400):
        di = _pick_layer(n, layer_pref)
        nd = g[di] - 1
        if abs(nd) > max_level or not _file_ok(path, names[di], nd):
            continue
        ii = _pick_layer(n, layer_pref)
        ni = g[ii] + 1
        if abs(ni) > max_level or not _file_ok(path, names[ii], ni):
            continue
        g2 = list(g)
        g2[di] = nd
        g2[ii] = ni
        if genome_cost(g2) <= max_dev:
            return g2
    return list(g)


def op_neutral_swap(g, names, path, max_level, max_dev, layer_pref=None):
    n = len(g)
    for _ in range(400):
        if layer_pref:
            nz = [i for i in layer_pref if g[i] != 0] or [i for i in range(n) if g[i] != 0]
        else:
            nz = [i for i in range(n) if g[i] != 0]
        if not nz:
            break
        src = random.choice(nz)
        new_src = g[src] - int(np.sign(g[src]))
        if not _file_ok(path, names[src], new_src):
            continue
        dst = _pick_layer(n, layer_pref)
        if dst == src:
            continue
        if g[dst] == 0:
            new_dst = random.choice([1, -1])
        else:
            new_dst = g[dst] + int(np.sign(g[dst]))
        if abs(new_dst) > max_level or not _file_ok(path, names[dst], new_dst):
            continue
        g2 = list(g); g2[src] = new_src; g2[dst] = new_dst
        if genome_cost(g2) <= max_dev:
            return g2
    return _flip_pair(g, names, path, max_level, max_dev, layer_pref)


def op_walk_1(g, names, path, max_level, max_dev, layer_pref=None):
    return _flip_pair(g, names, path, max_level, max_dev, layer_pref)

def op_walk_2(g, names, path, max_level, max_dev, layer_pref=None):
    g = _flip_pair(g, names, path, max_level, max_dev, layer_pref)
    return _flip_pair(g, names, path, max_level, max_dev, layer_pref)

def op_walk_3(g, names, path, max_level, max_dev, layer_pref=None):
    g = _flip_pair(g, names, path, max_level, max_dev, layer_pref)
    g = _flip_pair(g, names, path, max_level, max_dev, layer_pref)
    return _flip_pair(g, names, path, max_level, max_dev, layer_pref)


def op_rebalance(g, names, path, max_level, max_dev, layer_pref=None):
    n = len(g)
    pool = layer_pref if layer_pref else list(range(n))
    if not pool:
        return list(g)
    src = max(pool, key=lambda i: abs(g[i]))
    if g[src] == 0:
        return op_walk_1(g, names, path, max_level, max_dev, layer_pref)
    new_src = g[src] - int(np.sign(g[src]))
    if not _file_ok(path, names[src], new_src):
        return op_walk_1(g, names, path, max_level, max_dev, layer_pref)
    for dst in sorted(pool, key=lambda i: abs(g[i])):
        if dst == src:
            continue
        new_dst = (g[dst] + int(np.sign(g[dst]))) if g[dst] != 0 else random.choice([1, -1])
        if abs(new_dst) <= max_level and _file_ok(path, names[dst], new_dst):
            g2 = list(g); g2[src] = new_src; g2[dst] = new_dst
            if genome_cost(g2) <= max_dev:
                return g2
    return op_walk_1(g, names, path, max_level, max_dev, layer_pref)


def op_crossover(g1, g2, names, path, max_level, max_dev, layer_pref=None):
    for _ in range(200):
        child = [g1[i] if random.random() < 0.5 else g2[i] for i in range(len(g1))]
        if genome_cost(child) <= max_dev:
            return child
    child = [g1[i] if random.random() < 0.5 else g2[i] for i in range(len(g1))]
    while genome_cost(child) > max_dev:
        nz = [i for i in range(len(child)) if child[i] != 0]
        if not nz:
            break
        i = max(nz, key=lambda k: abs(child[k]))
        new_lv = child[i] - int(np.sign(child[i]))
        child[i] = new_lv if _file_ok(path, names[i], new_lv) else 0
    return child


OPERATORS = ["neutral_swap", "walk_1", "walk_2", "walk_3", "rebalance"]
OP_FNS = {
    "neutral_swap": op_neutral_swap,
    "walk_1": op_walk_1, "walk_2": op_walk_2, "walk_3": op_walk_3,
    "rebalance": op_rebalance,
}


# ─────────────────────────────────────────────────────────────────────────────
# Basin priors: derive bandit weights from basin descriptor centroid
# ─────────────────────────────────────────────────────────────────────────────

def bandit_for_basin(basin: BasinSpec, gamma: float = 0.07) -> EXP3Bandit:
    """
    Bias bandit weights by the basin's descriptor centroid.
    Higher layer_entropy → favour neutral_swap (fine-grain). Lower entropy
    (concentrated budget) → favour walk_2/walk_3 (coarse moves).
    """
    b = EXP3Bandit(OPERATORS, gamma=gamma)
    # centroid = (attn_share, mlp_share, early_share, late_share, entropy, positive_share)
    ent = basin.descriptor_centroid[4] if len(basin.descriptor_centroid) > 4 else 0.5
    if ent >= 0.6:
        b.w.update({"neutral_swap": 3.0, "walk_1": 2.5, "rebalance": 1.5, "walk_2": 1.0, "walk_3": 0.3})
    elif ent >= 0.35:
        b.w.update({"walk_1": 2.0, "walk_2": 2.0, "neutral_swap": 1.5, "rebalance": 1.5, "walk_3": 1.0})
    else:
        b.w.update({"walk_3": 3.0, "walk_2": 2.5, "rebalance": 1.5, "walk_1": 0.5, "neutral_swap": 0.5})
    return b


def basin_layer_pref(basin: BasinSpec, layer_groups: Dict[str, List[int]]) -> Optional[List[int]]:
    """Derive a layer-preference set from the basin's attn/mlp/early/late shares."""
    attn_s, mlp_s, early_s, late_s = basin.descriptor_centroid[:4]
    bias = max([("attn", attn_s), ("mlp", mlp_s), ("early", early_s), ("late", late_s)],
               key=lambda x: x[1])
    if bias[1] >= 0.55:
        return layer_groups[bias[0]]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────────────────────

def compute_fitness(model, data, fitness_fn, target_logits=None) -> float:
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    return compute_kl_div(model, data, target_logits)


def eval_pool(model, layer_names, weights_path, pool, calib_data,
              num_tokens, fitness_fn, target_logits) -> List[float]:
    mb, lmb = sample_minibatch(calib_data, num_tokens, target_logits if fitness_fn == "kl" else None)
    fitnesses = []
    for g in pool:
        load_layers(model, layer_names, g, weights_path)
        fitnesses.append(compute_fitness(model, mb, fitness_fn, lmb))
    return fitnesses


# ─────────────────────────────────────────────────────────────────────────────
# Island generation
# ─────────────────────────────────────────────────────────────────────────────

def island_generation(
    island: Island,
    layer_pref: Optional[List[int]],
    descriptor_fn,
    model, layer_names, weights_path, calib_data, target_logits,
    max_level: int, max_dev: int,
    n_offspring: int, pop_size: int,
    survivors_per_selection, tokens_per_selection,
    fitness_fn: str, hof: HallOfFame,
    crossover_rate: float = 0.40,
):
    population = island.pop
    bandit = island.bandit
    basin = island.basin

    offspring: List[List[int]] = []
    op_log: List[Tuple[str, int]] = []
    seen = {tuple(p) for p in population}
    op_counts = {op: 0 for op in OPERATORS + ["crossover"]}

    attempts = 0
    while len(offspring) < n_offspring and attempts < n_offspring * 30:
        attempts += 1
        if len(population) >= 2 and random.random() < crossover_rate:
            p1, p2 = random.sample(population, 2)
            child = op_crossover(p1, p2, layer_names, weights_path,
                                  max_level, max_dev, layer_pref)
            op = bandit.sample()
            child = OP_FNS[op](child, layer_names, weights_path,
                               max_level, max_dev, layer_pref)
            op_counts["crossover"] += 1
        else:
            parent = random.choice(population)
            op = bandit.sample()
            child = OP_FNS[op](parent, layer_names, weights_path,
                               max_level, max_dev, layer_pref)
        # Basin repair: if outside the descriptor radius, try once more
        if not in_basin(descriptor_fn(child), basin):
            op2 = bandit.sample()
            cand = OP_FNS[op2](child, layer_names, weights_path,
                               max_level, max_dev, layer_pref)
            if in_basin(descriptor_fn(cand), basin):
                child = cand
                op = op2
        sig = tuple(child)
        if sig in seen:
            continue
        seen.add(sig)
        op_log.append((op, len(offspring)))
        offspring.append(child)
        op_counts[op] += 1

    elite_pool = [p for p in population if tuple(p) not in {tuple(c) for c in offspring}]
    full_pool = offspring + elite_pool

    stage_survivors = list(survivors_per_selection[:-1]) + [pop_size]
    pool = list(full_pool)
    final_fitnesses = [float("inf")] * len(pool)
    for n_surv, n_tok in zip(stage_survivors, tokens_per_selection):
        n_surv = min(n_surv, len(pool))
        fitnesses = eval_pool(model, layer_names, weights_path, pool,
                              calib_data, n_tok, fitness_fn, target_logits)
        best_ids = list(np.argsort(fitnesses)[:n_surv])
        pool = [pool[i] for i in best_ids]
        final_fitnesses = [fitnesses[i] for i in best_ids]

    for g, f in zip(pool, final_fitnesses):
        hof.try_add(g, f)

    survivor_sigs = {tuple(s) for s in pool}
    for op, cidx in op_log:
        if cidx < len(offspring):
            reward = 1.0 if tuple(offspring[cidx]) in survivor_sigs else 0.0
            bandit.update(op, reward)

    return pool, (final_fitnesses[0] if final_fitnesses else float("inf")), op_counts


def warm_restart_island(island: Island, hof: HallOfFame, layer_names,
                         weights_path, max_level, max_dev, layer_pref,
                         layer_groups=None):
    """
    Diversifying restart:
      slot 0       : keep current island best (elite preservation)
      next ~1/3    : HoF seeds (exploit good basins)
      remaining    : FRESH random feasible genomes with varied biases
    Also resets bandit weights so locked-in operators get re-explored.
    """
    pop_size = len(island.pop)
    if pop_size == 0:
        return
    new_pop = [list(island.pop[0])]

    n_hof = max(0, pop_size // 3)
    if n_hof > 0 and len(hof) > 0:
        for seed in hof.sample_seeds(n_hof):
            new_pop.append(list(seed))

    biases = ["uniform", "attn", "mlp", "early", "late"]
    while len(new_pop) < pop_size:
        bias = random.choice(biases)
        g = sample_random_genome(
            layer_names, weights_path, max_level, max_dev,
            available_levels=[], bias=bias, layer_groups=layer_groups,
        )
        new_pop.append(g)

    island.pop = new_pop
    for op in island.bandit.ops:
        island.bandit.w[op] = 1.0
    island.stag = 0
    island.best_fitness = float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Args
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
    p.add_argument("--sparse_weights_path", required=True, type=str)
    p.add_argument("--max_level", default=99999, type=int)
    p.add_argument("--max_total_deviation", default=99999, type=int)
    p.add_argument("--generations", required=True, type=int)
    p.add_argument("--offspring", required=True, type=int)
    p.add_argument("--survivors_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--tokens_per_selection", nargs="+", type=int, required=True)
    p.add_argument("--num_islands", type=int, default=4, help="CVT cells = islands")
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
    p.add_argument("--configuration_name", type=str, default="final_configuration.txt")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    fix_seed(args.seed)
    if args.log_wandb:
        assert has_wandb, "wandb not installed"
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
        model.config.max_position_embeddings, 8192
    )
    calib_data = get_data(args.calibration_data, args.calibration_tokens,
                          args.calibration_sequence_length, tokenizer, train=True)

    args.eval_sequence_length = args.eval_sequence_length or min(
        model.config.max_position_embeddings, 8192
    )
    eval_datasets = [get_data(n, args.eval_tokens, args.eval_sequence_length, tokenizer, train=False)
                     for n in args.eval_datasets]

    target_logits = []
    if args.fitness_fn == "kl":
        print("Computing target logits...")
        for i in trange(len(calib_data), desc="Target logits", leave=False):
            with torch.no_grad():
                target_logits.append(model(calib_data[i].to(device)).logits.cpu())

    layer_names = sorted(
        n for n in os.listdir(args.sparse_weights_path)
        if os.path.isdir(os.path.join(args.sparse_weights_path, n))
    )
    n_layers = len(layer_names)
    layer_groups = classify_layers(layer_names)
    print(f"Found {n_layers} layers — {len(layer_groups['attn'])} attn, {len(layer_groups['mlp'])} mlp.")

    baseline = [0] * n_layers
    model.state = [None] * n_layers

    descriptor_fn = make_descriptor_fn(layer_groups)
    bounds_fn = make_bounds_fn()

    # ── CVT: sample 300 valid genomes, cluster to N basins ────────────────────
    print(f"Sampling {args.cvt_samples} random feasible genomes for CVT...")
    biases = ["uniform", "attn", "mlp", "early", "late"]
    samples = []
    for i in trange(args.cvt_samples, desc="CVT sampling", leave=False):
        bias = biases[i % len(biases)]
        g = sample_random_genome(layer_names, args.sparse_weights_path,
                                  args.max_level, args.max_total_deviation,
                                  available_levels=[], bias=bias, layer_groups=layer_groups)
        samples.append(g)

    basins = derive_basins_from_samples(
        samples=samples,
        descriptor_fn=descriptor_fn,
        bounds_fn=bounds_fn,
        k=args.num_islands,
        iters=args.cvt_iters,
        name_prefix="prune",
        seed=args.seed,
    )
    print(f"CVT produced {len(basins)} basins:")
    for b in basins:
        print(f"  {b.name}  mass={b.descriptor_mass}  centroid={tuple(round(c,2) for c in b.descriptor_centroid)}  r={b.descriptor_radius:.2f}")

    # ── Build islands ─────────────────────────────────────────────────────────
    islands = []
    island_layer_prefs = []
    for basin in basins:
        bandit = bandit_for_basin(basin)
        layer_pref = basin_layer_pref(basin, layer_groups)
        # Seed each island's population from samples landing nearest its centroid
        nearest = sorted(samples, key=lambda g: math.sqrt(sum(
            ((d - c) / max(s, 1e-6)) ** 2
            for d, c, s in zip(descriptor_fn(g), basin.descriptor_centroid, basin.descriptor_scale)
        )))[:args.pop_size]
        pop = [list(g) for g in nearest] if nearest else [list(baseline) for _ in range(args.pop_size)]
        while len(pop) < args.pop_size:
            pop.append(list(baseline))
        islands.append(Island(name=basin.name, basin=basin, pop=pop,
                               bandit=bandit, stag_threshold=args.stag_threshold))
        island_layer_prefs.append(layer_pref)

    hof = HallOfFame(capacity=args.hof_capacity)
    temperature = CosineAnnealingTemperature(T_max=3.0, T_min=1.0,
                                              period=max(10, args.generations // 3))

    global_best_fitness = float("inf")
    global_best_genome = list(baseline)
    log_dict: Dict[str, Any] = {}

    print(f"\nRunning {args.generations} generations over {len(islands)} islands "
          f"(pop_size={args.pop_size}, offspring={args.offspring}).")

    for gen in trange(args.generations, desc="Generations"):
        T = temperature.step()

        for island, layer_pref in zip(islands, island_layer_prefs):
            new_pop, best_f, op_counts = island_generation(
                island=island, layer_pref=layer_pref, descriptor_fn=descriptor_fn,
                model=model, layer_names=layer_names, weights_path=args.sparse_weights_path,
                calib_data=calib_data, target_logits=target_logits if target_logits else None,
                max_level=args.max_level, max_dev=args.max_total_deviation,
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
                global_best_genome = list(new_pop[0])

        log_dict["train_fitness/global_best"] = global_best_fitness
        log_dict["temperature"] = T

        # Cross-pollinating migration (2 slots from any other island)
        if (gen + 1) % args.migration_every == 0:
            bests = [list(isl.pop[0]) for isl in islands]
            for i, isl in enumerate(islands):
                others = [bests[j] for j in range(len(islands)) if j != i]
                random.shuffle(others)
                present = {tuple(p) for p in isl.pop}
                n_replace = min(2, len(isl.pop) - 1, len(others))
                for k in range(n_replace):
                    donor = others[k]
                    if tuple(donor) not in present:
                        isl.pop[-(k + 1)] = donor
                        present.add(tuple(donor))
            msg = "  ".join(f"{isl.name}:{isl.best_fitness:.3e}" for isl in islands)
            print(f"\n[Gen {gen+1}] migrate — {msg}")

        # Stagnation warm restarts (with fresh randoms)
        for island, layer_pref in zip(islands, island_layer_prefs):
            if island.is_stagnant():
                print(f"\n[Gen {gen+1}] warm-restart {island.name} (stag={island.stag})")
                warm_restart_island(island, hof, layer_names, args.sparse_weights_path,
                                     args.max_level, args.max_total_deviation, layer_pref,
                                     layer_groups=layer_groups)

        if args.restart_every > 0 and (gen + 1) % args.restart_every == 0:
            for island, layer_pref in zip(islands, island_layer_prefs):
                warm_restart_island(island, hof, layer_names, args.sparse_weights_path,
                                     args.max_level, args.max_total_deviation, layer_pref,
                                     layer_groups=layer_groups)
            print(f"\n[Gen {gen+1}] scheduled restart with fresh randoms")

        if gen % args.eval_every == 0:
            load_layers(model, layer_names, global_best_genome, args.sparse_weights_path)
            for dname, dset in zip(args.eval_datasets, eval_datasets):
                ppl = compute_perplexity(model, dset)
                print(f"  [{dname}] ppl={ppl:.3f}")
                log_dict[f"ppl_eval/{dname}"] = ppl
            ppl_train = compute_perplexity(model, calib_data)
            print(f"  [train] ppl={ppl_train:.3f}  fit={global_best_fitness:.3e}  T={T:.2f}")
            log_dict["ppl_train"] = ppl_train
            log_dict["gen"] = gen
            for island in islands:
                print(f"  [{island.name}] {island.bandit.summary()}")

        if args.log_wandb:
            wandb.log(log_dict)

    # Final evaluation
    hof_g, hof_f = hof.best()
    if hof_g is not None and hof_f < global_best_fitness:
        global_best_genome = hof_g
        global_best_fitness = hof_f

    load_layers(model, layer_names, global_best_genome, args.sparse_weights_path)
    print("\n=== Final ===")
    for dname, dset in zip(args.eval_datasets, eval_datasets):
        ppl = compute_perplexity(model, dset)
        print(f"  {dname}: {ppl:.3f}")
        log_dict[f"ppl_eval_final/{dname}"] = ppl

    out_path = os.path.join(args.sparse_weights_path, args.configuration_name)
    with open(out_path, "w") as f:
        f.write("\n".join(f"{n}: {lv}" for n, lv in zip(layer_names, global_best_genome)))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
