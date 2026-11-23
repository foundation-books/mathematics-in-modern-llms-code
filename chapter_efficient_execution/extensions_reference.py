"""Typed unit-slot training schedules with versioned saved-state consumers."""
from dataclasses import dataclass
import sympy as sp

R = sp.Rational
# Stage cells determine resource ownership, not merely how jobs are displayed.
IMMEDIATE = (
    (('F', 0, 0), None), (('F', 0, 1), ('F', 1, 0)), (None, ('D', 1, 0)),
    (('D', 0, 0), ('W', 1, 0)), (('W', 0, 0), ('F', 1, 1)),
    (None, ('D', 1, 1)), (('D', 0, 1), ('W', 1, 1)), (('W', 0, 1), None))
DEFERRED = (
    (('F', 0, 0), None), (('F', 0, 1), ('F', 1, 0)), (None, ('D', 1, 0)),
    (('D', 0, 0), ('F', 1, 1)), (('W', 0, 0), ('D', 1, 1)),
    (('D', 0, 1), ('W', 1, 0)), (('W', 0, 1), ('W', 1, 1)))
REQUIRED = frozenset((kind, stage, micro) for kind in 'FDW'
                     for stage in range(2) for micro in range(2))


@dataclass(frozen=True)
class SavedRecord:
    stage: int
    microbatch: int
    version: str = 'v0'


def check_schedule(rows, *, record_refs=None, job_versions=None,
                   release_at=None, optimizer_at=None, immediate_policy=False):
    """Validate the printed two-stage, two-microbatch, zero-transfer-time model.

    release_at gives explicit early release requests at slot-end times. A legal
    request can only occur after both local consumers; automatic release is the
    same rule. Record refs permit deliberate identity/version mutants.
    optimizer_at is a slot boundary; the update must wait for every W.
    """
    record_refs, job_versions = record_refs or {}, job_versions or {}
    release_at = release_at or {}
    if optimizer_at is not None and (type(optimizer_at) is not int or
                                     not 0 <= optimizer_at <= len(rows)):
        raise ValueError('optimizer boundary outside schedule')
    if any(type(t) is not int or not 1 <= t <= len(rows) for t in release_at):
        raise ValueError('release boundary outside schedule')
    done, live, times = set(), {}, {}
    ledger = [(0, 0)]
    work = 0
    for time, row in enumerate(rows):
        if optimizer_at == time and not all(j in done for j in REQUIRED if j[0] == 'W'):
            raise ValueError('optimizer admitted before complete gradient')
        if len(row) != 2:
            raise ValueError('one cell for each of two stages required')
        pending = []
        for stage, raw_job in enumerate(row):
            if raw_job is None:
                continue
            job = tuple(raw_job)
            if len(job) != 3 or any(type(v) is not int for v in job[1:]):
                raise ValueError('typed identity required')
            kind, owner, micro = job
            if job not in REQUIRED or owner != stage or job in done:
                raise ValueError('unknown, duplicated or wrongly owned job')
            if job_versions.get(job, 'v0') != 'v0':
                raise ValueError('frozen parameter version changed')
            deps = set()
            if kind == 'F' and stage > 0:
                deps.add(('F', stage - 1, micro))
            if kind in 'DW':
                deps.add(('F', stage, micro))
                if stage < 1:
                    deps.add(('D', stage + 1, micro))
                expected = SavedRecord(stage, micro)
                ref = record_refs.get(job, expected)
                if ref != expected or live.get((stage, micro)) != expected:
                    raise ValueError('missing, stale or mismatched saved record')
                if kind == 'W' and immediate_policy and times.get(('D', stage, micro)) != time - 1:
                    raise ValueError('immediate local D/W policy violated')
            if not deps <= done:
                raise ValueError('dependency not complete at slot start')
            pending.append(job)
        # All checks above used the same slot-start done set.
        for job in pending:
            kind, stage, micro = job
            done.add(job)
            times[job] = time
            if kind == 'F':
                live[(stage, micro)] = SavedRecord(stage, micro)
        for key in release_at.get(time + 1, ()):
            stage, micro = key
            if key not in live or not {('D', stage, micro), ('W', stage, micro)} <= done:
                raise ValueError('saved state released before its last consumer')
        live = {key: value for key, value in live.items()
                if not {('D', key[0], key[1]), ('W', key[0], key[1])} <= done}
        ledger.append(tuple(sum(st == stage for st, _ in live) for stage in range(2)))
        work += len(pending)
    if done != REQUIRED:
        raise ValueError('missing complete-step work')
    if optimizer_at == len(rows) and not all(j in done for j in REQUIRED if j[0] == 'W'):
        raise ValueError('incomplete optimizer barrier')
    return dict(makespan=len(rows), useful_jobs=work,
                idle_fraction=R(2 * len(rows) - work, 2 * len(rows)),
                peaks=tuple(max(row[s] for row in ledger) for s in range(2)),
                ledger=tuple(ledger))


def execute(rows):
    """Execute forward and both local backward products, then one SGD step.

    The independent test oracle differentiates the complete loss instead of
    reusing this task executor's local formulas.
    """
    stats = check_schedule(rows, optimizer_at=len(rows))
    a, b, rate = R(1, 2), sp.Integer(2), R(1, 100)
    xs, targets = (4, 5), (1, 2)
    h, y, gy, gh, gx = {}, {}, {}, {}, {}
    contributions = {0: {}, 1: {}}
    for row in rows:
        for job in row:
            if job is None:
                continue
            kind, stage, i = job
            if kind == 'F':
                if stage == 0:
                    h[i] = a * xs[i]
                else:
                    y[i] = b * h[i]
                    gy[i] = (y[i] - targets[i]) / 2  # one global mean denominator
            elif kind == 'D':
                if stage == 1:
                    gh[i] = b * gy[i]
                else:
                    gx[i] = a * gh[i]
            elif stage == 1:
                contributions[1][i] = h[i] * gy[i]
            else:
                contributions[0][i] = xs[i] * gh[i]
    gradients = tuple(sum(contributions[s].values()) for s in range(2))
    return dict(stats=stats, hidden=tuple(h[i] for i in range(2)),
                outputs=tuple(y[i] for i in range(2)),
                output_seeds=tuple(gy[i] for i in range(2)),
                input_seeds=tuple(gx[i] for i in range(2)),
                contributions=tuple(tuple(contributions[s][i] for i in range(2)) for s in range(2)),
                loss=sum((y[i] - targets[i]) ** 2 for i in range(2)) / 4,
                gradients=gradients,
                next_parameters=(a - rate * gradients[0], b - rate * gradients[1]))


if __name__ == '__main__':
    for name, rows in [('I', IMMEDIATE), ('D', DEFERRED)]:
        result = execute(rows)
        print(name, 'gradients', result['gradients'], 'next', result['next_parameters'])
        print(name, result['stats'])
    print('Unit model only: no measured GPU utilization or total GPU-memory claim.')
