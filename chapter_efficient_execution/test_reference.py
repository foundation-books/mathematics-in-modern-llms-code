"""Behavioral regression checks for tiled attention and numerical comparison."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import pytest
import sympy as sp

spec = spec_from_file_location('chapter21_reference', Path(__file__).with_name('reference.py'))
ref = module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)


def test_complete_independent_assessment_oracles():
    report = ref.run_checks()
    assert report['finite_sweep'] == {
        'paired_permutations': 6, 'binary_trees': 12, 'chunkings': 24,
        'tile_shapes': 9, 'allowed_pairs_per_shape': 6,
        'causality_perturbation_shapes': 9}
    assert len(report['deliberate_fault_witnesses']) == 3
    assert report['C_resources']['tiles'] == {
        '2x2': {'fast_memory_bytes': 72, 'fits_96_bytes': True, 'no_spill_traffic_bytes': 320},
        '4x2': {'fast_memory_bytes': 128, 'fits_96_bytes': False, 'no_spill_traffic_bytes': 192},
    }
    assert len(report['D_working_precision']) == 10


def test_empty_intermediate_is_identity_but_empty_output_rejects():
    s, v, *_ = ref.fixture()
    state = ref.summarize((0,), s, v)
    assert ref.merge(None, state) is state
    assert ref.merge(state, None) is state
    assert ref.merge(None, None) is None
    with pytest.raises(ValueError):
        ref.finalize(None)


def test_signed_nonfixture_width_and_duplicate_multiplicity():
    s = (sp.Integer(0), sp.log(2))
    v = sp.Matrix([[3, -1, 4], [-2, 5, 0]])
    state = ref.merge(ref.summarize((0,), s, v), ref.summarize((1,), s, v))
    assert ref.finalize(state) == ref.reference_row((0, 1), s, v)[1]
    duplicate = ref.merge(state, ref.summarize((0,), s, v))
    assert ref.finalize(duplicate) == sp.Matrix([sp.Rational(1, 2), 2, 2])
    assert ref.finalize(duplicate) != ref.finalize(state)


@pytest.mark.parametrize('bad', [sp.oo, sp.nan, sp.I, sp.Symbol('unknown')])
def test_admissible_domain_requires_finite_real_scores(bad):
    with pytest.raises(ValueError):
        ref.summarize((0,), (bad,), sp.Matrix([[1, 0]]))


def test_shapes_masks_and_disjoint_coverage_are_enforced():
    s, v, _, k, q = ref.fixture()
    with pytest.raises(ValueError):
        ref.reference_row((0, 0), s, v)
    with pytest.raises(ValueError):
        ref.summarize((-1,), s, v)
    with pytest.raises(ValueError):
        ref.distributed_row(((0, 1), (1, 2)), s, v)
    with pytest.raises(ValueError):
        ref.distributed_row(((0,), (1,)), s, v)
    with pytest.raises(ValueError):
        ref.tiled_attention(q, k, v, [(), (0,), (0,)], 2, 4)
    with pytest.raises(ValueError):
        ref.tiled_attention(q, k[:, :1], v, [(0,)]*3, 2, 4)


def test_tile_summaries_receive_only_current_key_block(monkeypatch):
    _, values, _, keys, queries = ref.fixture()
    widths = []
    summarize = ref.summarize

    def observe(ids, scores, value_rows):
        assert len(scores) == value_rows.rows
        widths.append(len(scores))
        return summarize(ids, scores, value_rows)

    monkeypatch.setattr(ref, 'summarize', observe)
    ref.tiled_attention(queries, keys, values, [(0,), (0, 1), (0, 1, 2)], 2, 2)
    assert widths == [2, 2, 1, 1, 2, 1]


@pytest.mark.parametrize('queries,keys', [
    (sp.Matrix([[sp.I]]), sp.Matrix([[sp.I]])),
    (sp.Matrix([[sp.Symbol('unknown')]]), sp.Matrix([[0]])),
    (sp.Matrix([[0]]), sp.Matrix([[sp.Symbol('unknown')]])),
    (sp.Matrix([[sp.oo]]), sp.Matrix([[0]])),
])
def test_tiled_inputs_obey_the_domain_before_score_simplification(queries, keys):
    # Complex factors can produce a real score; zero can erase an unknown input.
    # Neither cancellation makes the original Q/K admissible.
    with pytest.raises(ValueError, match='Attention inputs'):
        ref.tiled_attention(queries, keys, sp.Matrix([[1]]), [(0,)], 1, 1)


def test_normwise_policy_is_global_and_rejects_nonfinite_values():
    assert ref.normwise_agreement([1000, 0.5], [1000, 0], 0, '0.001')
    assert not ref.normwise_agreement([sp.oo], [0], 1, 1)
    assert not ref.normwise_agreement([1], [0], 0, '0.01')
    assert ref.normwise_agreement(['1e-12'], [0], '1e-11', 0)
    with pytest.raises(ValueError):
        ref.normwise_agreement([0], [0], -1, 0)
