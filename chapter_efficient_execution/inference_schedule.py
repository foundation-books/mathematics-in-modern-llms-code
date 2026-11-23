"""Efficient execution exercise 6(d): exact schedules for six unit jobs."""
from itertools import product
import json
import sympy as sp

SCHEDULES = {
    1: (('G1',), ('C1',), ('G2',), ('C2',), ('G3',), ('C3',)),
    2: (('G1', 'G2'), ('C1', 'G3'), ('C2', 'C3')),
    3: (('G1', 'G2', 'G3'), ('C1', 'C2', 'C3')),
}
JOBS = frozenset(f'{kind}{i}' for kind in 'GC' for i in (1, 2, 3))


def validate_schedule(slots, workers):
    """Each unit job once, capacity respected, each G_i strictly before C_i."""
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('Use a positive worker count.')
    done = set()
    for slot in slots:
        if not slot or len(slot) > workers or len(set(slot)) != len(slot):
            raise ValueError('A slot violates nonpreemptive worker capacity.')
        for job in slot:
            if job not in JOBS or job in done:
                raise ValueError('Unknown or repeated logical job.')
            if job[0] == 'C' and 'G'+job[1] not in done:
                raise ValueError('A check must follow its generation.')
        done.update(slot)
    if done != JOBS:
        raise ValueError('All six jobs execute; no cancellation.')
    return len(slots)


def selected_index(draws):
    """Three logical draws, class 0 correct; exact checks, lowest index wins."""
    return next((i for i, value in enumerate(draws) if value == 0), None)


def run_checks():
    w, depth = 6, 2
    times = {}
    for workers, slots in SCHEDULES.items():
        elapsed = validate_schedule(slots, workers)
        assert elapsed == max(sp.Rational(w, workers), depth)
        times[workers] = elapsed
    q = (sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 6))
    correct, abstain = sp.Integer(0), sp.Integer(0)
    for draws in product(range(3), repeat=3):
        mass = sp.prod(q[i] for i in draws)
        outputs = []
        for slots in SCHEDULES.values():
            checked = {}
            for slot in slots:
                for job in slot:
                    if job[0] == 'C':
                        i = int(job[1])-1
                        checked[i] = draws[i] == 0
            outputs.append(next((i for i in range(3) if checked[i]), None))
        assert len(set(outputs)) == 1
        assert outputs[0] == selected_index(draws)
        if outputs[0] is None:
            abstain += mass
        else:
            correct += mass
    assert (correct, abstain) == (sp.Rational(7, 8), sp.Rational(1, 8))
    return dict(work=w, dependency_depth=depth, times=times, ordered_pools=27,
                correct=correct, wrong=0, abstain=abstain)


if __name__ == '__main__':
    print(json.dumps(run_checks(), default=str, indent=2))
