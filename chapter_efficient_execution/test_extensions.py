import unittest
import sympy as sp
import extensions_reference as pipe

R = sp.Rational


def changed(rows):
    return [list(row) for row in rows]


class PipelineTests(unittest.TestCase):
    def test_independent_global_loss_and_both_executions(self):
        a, b = sp.symbols('a b', real=True)
        loss = ((b * a * 4 - 1) ** 2 + (b * a * 5 - 2) ** 2) / 4
        at = {a: R(1, 2), b: 2}
        gradients = tuple(sp.diff(loss, p).subs(at) for p in [a, b])
        self.assertEqual(gradients, (27, R(27, 4)))
        for rows in [pipe.IMMEDIATE, pipe.DEFERRED]:
            result = pipe.execute(rows)
            self.assertEqual(result['loss'], loss.subs(at))
            self.assertEqual(result['hidden'], (2, R(5, 2)))
            self.assertEqual(result['outputs'], (4, 5))
            self.assertEqual(result['output_seeds'], (R(3, 2), R(3, 2)))
            self.assertEqual(result['input_seeds'], (R(3, 2), R(3, 2)))
            self.assertEqual(result['contributions'], ((12, 15), (3, R(15, 4))))
            self.assertEqual(result['gradients'], gradients)
            self.assertEqual(result['next_parameters'], (R(23, 100), R(773, 400)))

    def test_full_saved_state_ledgers_and_window_costs(self):
        i = pipe.check_schedule(pipe.IMMEDIATE, immediate_policy=True)
        d = pipe.check_schedule(pipe.DEFERRED)
        self.assertEqual((i['makespan'], d['makespan']), (8, 7))
        self.assertEqual((i['useful_jobs'], d['useful_jobs']), (12, 12))
        self.assertEqual((i['idle_fraction'], d['idle_fraction']), (R(1, 4), R(1, 7)))
        self.assertEqual((i['peaks'], d['peaks']), ((2, 1), (2, 2)))
        # Independent lifetime intervals obtained from the printed timeline.
        for rows, result in [(pipe.IMMEDIATE, i), (pipe.DEFERRED, d)]:
            ends = {job: t + 1 for t, row in enumerate(rows) for job in row if job}
            oracle = []
            for t in range(len(rows) + 1):
                oracle.append(tuple(sum(ends['F', st, mb] <= t < max(ends['D', st, mb], ends['W', st, mb])
                                        for mb in range(2)) for st in range(2)))
            self.assertEqual(result['ledger'], tuple(oracle))
        self.assertEqual(tuple(row[1] for row in d['ledger']), (0, 0, 1, 1, 2, 2, 1, 0))

    def test_finite_dependency_model_has_no_schedule_shorter_than_seven(self):
        # Independent breadth-first state search: all ready stage-job pairs.
        required = {(k, st, mb) for k in 'FDW' for st in range(2) for mb in range(2)}
        def ready(job, done):
            kind, st, mb = job
            deps = set()
            if kind == 'F' and st == 1:
                deps.add(('F', 0, mb))
            if kind in 'DW':
                deps.add(('F', st, mb))
                if st == 0:
                    deps.add(('D', 1, mb))
            return deps <= done
        states = {frozenset()}
        for depth in range(8):
            if frozenset(required) in states:
                self.assertEqual(depth, 7)
                return
            following = set()
            for done in states:
                choices = [[None] + [j for j in required - set(done) if j[1] == st and ready(j, done)] for st in range(2)]
                for first in choices[0]:
                    for second in choices[1]:
                        following.add(done | frozenset(j for j in [first, second] if j))
            states = following
        self.fail('Complete step not reached within the proved seven-slot bound')

    def test_premature_and_same_slot_dependencies(self):
        rows = changed(pipe.DEFERRED)
        rows[2][0] = ('D', 0, 0)  # successor D has not completed at slot start
        with self.assertRaises(ValueError):
            pipe.check_schedule(rows)
        rows = changed(pipe.DEFERRED)
        rows[0][1] = ('F', 1, 0)  # same-slot producer is not completed
        with self.assertRaises(ValueError):
            pipe.check_schedule(rows)

    def test_missing_duplicate_wrong_owner_and_identity(self):
        rows = changed(pipe.DEFERRED)
        rows[6][1] = None
        with self.assertRaises(ValueError):
            pipe.check_schedule(rows)
        with self.assertRaises(ValueError):
            pipe.check_schedule(list(pipe.DEFERRED) + [(None, ('W', 1, 1))])
        rows = changed(pipe.DEFERRED)
        rows[0][0] = ('F', 1, 0)
        with self.assertRaises(ValueError):
            pipe.check_schedule(rows)
        rows = changed(pipe.DEFERRED)
        rows[0][0] = ('F', 0, 2)
        with self.assertRaises(ValueError):
            pipe.check_schedule(rows)

    def test_record_microbatch_version_and_release(self):
        with self.assertRaises(ValueError):
            pipe.check_schedule(pipe.DEFERRED, record_refs={('W', 1, 0): pipe.SavedRecord(1, 1)})
        with self.assertRaises(ValueError):
            pipe.check_schedule(pipe.DEFERRED, record_refs={('W', 1, 0): pipe.SavedRecord(1, 0, 'stale')})
        with self.assertRaises(ValueError):
            pipe.check_schedule(pipe.DEFERRED, job_versions={('D', 0, 1): 'v1'})
        with self.assertRaises(ValueError):
            pipe.check_schedule(pipe.DEFERRED, release_at={3: [(1, 0)]})
        # Release at last consumer is legal; it must not change the ledger.
        self.assertEqual(pipe.check_schedule(pipe.DEFERRED, release_at={6: [(1, 0)]}),
                         pipe.check_schedule(pipe.DEFERRED))

    def test_optimizer_barrier_and_early_update_changes_function(self):
        with self.assertRaises(ValueError):
            pipe.check_schedule(pipe.DEFERRED, optimizer_at=4)
        self.assertEqual(pipe.check_schedule(pipe.DEFERRED, optimizer_at=7)['makespan'], 7)
        early_b = 2 - R(1, 100) * 3
        early_second_output = early_b * R(1, 2) * 5
        self.assertEqual((early_b, early_second_output), (R(197, 100), R(197, 40)))
        self.assertNotEqual(early_second_output, 5)

    def test_mathematical_readiness_versus_optional_immediate_policy(self):
        # Stage1 W00 can precede D10: both need the same saved input/output seed.
        alternative = changed(pipe.IMMEDIATE)
        alternative[2][1], alternative[3][1] = alternative[3][1], alternative[2][1]
        # Stage0 D00 must now wait one more slot, without sharing its W resource.
        alternative[3][0] = None
        alternative[4][0] = ('D', 0, 0)
        alternative[5][0] = ('W', 0, 0)
        result = pipe.execute(alternative)
        self.assertEqual(result['gradients'], (27, R(27, 4)))
        with self.assertRaises(ValueError):
            pipe.check_schedule(alternative, immediate_policy=True)


if __name__ == '__main__':
    unittest.main()
