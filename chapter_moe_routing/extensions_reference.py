"""Exact instructor constructions for group selection and persistent replicas.

Logical expert/group IDs are positive integers. Physical and device IDs are
zero-based. No student starter is imported; no tensor/GPU library is required.
"""
from collections import Counter
from itertools import combinations
import sympy as sp

R = sp.Rational


def rational(value):
    if isinstance(value, (float, bool)):
        raise ValueError("exact finite rational required")
    value = sp.sympify(value)
    if value.is_Rational is not True:
        raise ValueError("exact finite rational required")
    return value


def integer(value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("integer outside declared domain")
    return value


def routing_inputs(scores, groups, k, g):
    scores = {e: rational(v) for e, v in scores.items()}
    if not scores or any(type(e) is not int or e < 1 for e in scores):
        raise ValueError("nonempty positive expert identities required")
    if not groups or any(type(j) is not int or j < 1 for j in groups):
        raise ValueError("positive group identities required")
    members = [e for group in groups.values() for e in group]
    if any(not group for group in groups.values()) or Counter(members) != Counter(scores.keys()):
        raise ValueError("groups must be a disjoint, complete, nonempty partition")
    if integer(k) > len(scores) or integer(g) > len(groups):
        raise ValueError("selection budget out of range")
    return scores


def topk(scores, eligible, k):
    eligible = tuple(eligible)
    if len(set(eligible)) != len(eligible) or len(eligible) < integer(k):
        raise ValueError("insufficient or repeated eligible identities")
    return tuple(sorted(sorted(eligible, key=lambda e: (-scores[e], e))[:k]))


def group_select(scores, groups, k, g):
    scores = routing_inputs(scores, groups, k, g)
    opened = sorted(groups, key=lambda j: (-max(scores[e] for e in groups[j]), j))[:g]
    return topk(scores, [e for j in opened for e in groups[j]], k)


def constrained_select(scores, groups, k, g):
    scores = routing_inputs(scores, groups, k, g)
    group_of = {e: j for j, members in groups.items() for e in members}
    feasible = [t for t in combinations(sorted(scores), k)
                if len({group_of[e] for e in t}) <= g]
    if not feasible:
        raise ValueError("empty constrained feasible family")
    return min(feasible, key=lambda t: (-sum(scores[e] for e in t), t))


def selected_mixture(selected, affinities, outputs):
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("nonempty distinct selected IDs required")
    raw = [rational(affinities[e]) for e in selected]
    if any(v <= 0 for v in raw):
        raise ValueError("positive raw affinities required")
    vectors = [sp.Matrix(outputs[e]) for e in selected]
    if len({v.shape for v in vectors}) != 1:
        raise ValueError("incompatible expert outputs")
    den = sum(raw)
    return sum((w * v for w, v in zip(raw, vectors)), sp.zeros(*vectors[0].shape)) / den


def work_values(weights):
    result = tuple(rational(w) for w in weights)
    if not result or any(w < 0 for w in result):
        raise ValueError("nonempty nonnegative work estimates required")
    return result


def replica_counts(weights, slots):
    weights = work_values(weights)
    if integer(slots) < len(weights):
        raise ValueError("insufficient replica slots")
    counts = [1] * len(weights)
    for _ in range(slots - len(weights)):
        e = min(range(len(weights)), key=lambda e: (-weights[e] / counts[e], e))
        counts[e] += 1
    return tuple(counts)


def threshold_feasible(weights, slots, tau):
    weights = work_values(weights)
    integer(slots, 0)
    tau = rational(tau)
    if tau < 0:
        raise ValueError("nonnegative threshold required")
    if tau == 0:
        return slots >= len(weights) and all(w == 0 for w in weights)
    return sum(max(1, int(sp.ceiling(w / tau))) for w in weights) <= slots


def packing(weights, counts, devices, capacities, *, device_nodes=None,
            allowed_nodes=None, sizes=None, memory=None, versions=None,
            required_version="v0"):
    """Build inverse physical/logical maps and idealized equal-split loads.

    devices[d] lists one-based logical IDs in distinct physical slots.
    Each entry occupies storage, including colocated copies of one expert.
    """
    weights = work_values(weights)
    counts = tuple(integer(r) for r in counts)
    if len(counts) != len(weights) or len(devices) != len(capacities) or not devices:
        raise ValueError("incompatible count/device dimensions")
    capacities = tuple(integer(c, 0) for c in capacities)
    if any(len(items) != cap for items, cap in zip(devices, capacities)):
        raise ValueError("exact slot capacities required")
    flat = tuple(e for items in devices for e in items)
    if any(type(e) is not int or not 1 <= e <= len(weights) for e in flat):
        raise ValueError("unknown logical expert")
    if Counter(flat) != Counter({e + 1: r for e, r in enumerate(counts)}):
        raise ValueError("replica counts do not match physical slots")
    if versions is not None and (len(versions) != len(flat) or
                                 any(v != required_version for v in versions)):
        raise ValueError("replicas must use the required logical parameter version")
    if allowed_nodes is not None:
        if device_nodes is None or len(device_nodes) != len(devices):
            raise ValueError("node map required")
        if any(device_nodes[d] not in allowed_nodes.get(e, set())
               for d, items in enumerate(devices) for e in items):
            raise ValueError("infeasible node placement")
    if (sizes is None) != (memory is None):
        raise ValueError("both byte sizes and memory budgets required")
    if sizes is not None:
        sizes = tuple(rational(v) for v in sizes)
        memory = tuple(rational(v) for v in memory)
        if len(sizes) != len(weights) or len(memory) != len(devices):
            raise ValueError("memory dimensions")
        if any(v <= 0 for v in sizes) or any(v < 0 for v in memory):
            raise ValueError("positive copy byte sizes and nonnegative byte budgets required")
        if any(sum(sizes[e - 1] for e in items) > memory[d]
               for d, items in enumerate(devices)):
            raise ValueError("parameter-copy byte budget exceeded")
    inverse = {e: tuple(p for p, logical in enumerate(flat) if logical == e)
               for e in range(1, len(weights) + 1)}
    loads = tuple(sum((weights[e - 1] / counts[e - 1] for e in items), sp.S.Zero)
                  for items in devices)
    return {"physical_to_logical": flat, "logical_to_physical": inverse,
            "loads": loads, "parameter_bytes": None if sizes is None else
            tuple(sum(sizes[e - 1] for e in items) for items in devices)}


def greedy_packs(weights, capacities):
    weights = work_values(weights)
    capacities = tuple(integer(c, 0) for c in capacities)
    if not capacities or sum(capacities) != len(weights):
        raise ValueError("capacities must account for all items")
    out = [[] for _ in capacities]
    for i in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
        d = min((d for d, c in enumerate(capacities) if len(out[d]) < c),
                key=lambda d: (sum(weights[j] for j in out[d]), d))
        out[d].append(i)
    return tuple(tuple(items) for items in out)


if __name__ == "__main__":
    scores = dict(enumerate([10, 1, 9, 8], 1))
    groups = {1: (1, 2), 2: (3, 4)}
    print("Group heuristic", group_select(scores, groups, 2, 1),
          "constrained optimum", constrained_select(scores, groups, 2, 1))
    print("Replica counts", replica_counts([70, 60, 50, 40], 8),
          "global packing", packing([70, 60, 50, 40], [2] * 4,
                                    [(1, 4), (1, 4), (2, 3), (2, 3)], [2] * 4)["loads"])
    print("Exact teaching constructions; no measured runtime or production-router claim.")
