"""Behavioral checks for probability laws and stopping, not implementation mirrors."""
from itertools import product
import importlib.util
from pathlib import Path
import sys

import pytest
import sympy as sp

spec = importlib.util.spec_from_file_location("ch22_reference", Path(__file__).with_name("reference.py"))
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
R = sp.Rational


def test_complete_joint_and_stale_prefix_witness():
    correct, branches = ref.pair_law()
    stale, _ = ref.pair_law(stale=True)
    assert correct == sp.diag(*ref.p)*ref.P
    assert sum(b[-1] for b in branches) == 1
    assert stale[0, 1] == R(5, 36)
    assert stale[0, 2] == R(1, 9)


@pytest.mark.parametrize("bad", [[-1, 2], [R(1, 3), R(1, 3)], [0.5, 0.5], [], [[1, 0]]])
def test_rejects_non_categorical_inputs(bad):
    with pytest.raises(ValueError):
        ref.categorical(bad)


def test_mismatched_vocabularies():
    with pytest.raises(ValueError):
        ref.verify([1, 0], [1, 0, 0])


def test_unreachable_residual_and_missing_support():
    assert ref.verify(ref.p, ref.p).residual is None
    result = ref.verify(ref.p, [1, 0, 0])
    assert result.residual == sp.Matrix([0, R(2, 3), R(1, 3)])
    assert result.output == ref.p


def test_threshold_equality_and_explicit_tie_order():
    assert ref.nucleus(ref.p, R(1, 2)) == sp.Matrix([1, 0, 0])
    assert ref.nucleus([R(1, 2), R(1, 2)], R(1, 2), (1, 0)) == sp.Matrix([0, 1])
    assert ref.nucleus(ref.softmax(ref.WT*ref.o_star), R(6, 7)) == sp.Matrix([
        R(1, 2), R(1, 3), 0, R(1, 6)])
    with pytest.raises(ValueError):
        ref.nucleus(ref.p, 0)


def test_selected_law_requires_changed_denominator():
    raw = sp.Matrix([R(1, 2), R(1, 2)])
    assert ref.verify(raw, raw).acceptance[0] == 1
    repaired = ref.verify(raw, [1, 0])
    assert repaired.acceptance[0] == R(1, 2)
    assert repaired.output == raw


@pytest.mark.parametrize("tokens,budget,expected", [
    (["EOS", "A", "B"], 3, ("EOS",)),
    (["A", "B", "EOS"], 1, ("A",)),
    (["A", "EOS", "B"], 3, ("A", "EOS")),
    (["A"], 0, ()),
])
def test_eos_and_budget_commit(tokens, budget, expected):
    assert ref.commit(tokens, budget) == expected


def test_full_stopped_laws_even_when_draft_never_proposes_eos():
    # Independent direct-target product supplies the expected law. Target
    # may stop immediately; q has incomplete support and never proposes EOS.
    def target(h):
        assert 1 not in h
        return sp.Matrix([R(1, 2), R(1, 2)]) if not h else sp.Matrix([0, 1])

    def proposal(h):
        assert 1 not in h
        return sp.Matrix([1, 0])

    for k, budget in product(range(1, 4), range(1, 4)):
        expected = {(0,): R(1, 2), (1,): R(1, 2)} if budget == 1 else {
            (0, 1): R(1, 2), (1,): R(1, 2)}
        assert ref.complete_law(target, proposal, k, budget, 1) == expected


def test_rejected_proposed_eos_does_not_terminate_output():
    def target(h):
        assert 1 not in h
        return sp.Matrix([1, 0]) if not h else sp.Matrix([0, 1])

    def proposal(h):
        assert 1 not in h
        return sp.Matrix([0, 1])

    assert ref.complete_law(target, proposal, 3, 3, 1) == {(0, 1): 1}


@pytest.mark.parametrize("k,budget", [(0, 1), (-1, 1), (1, -1), (True, 1), (1, True)])
def test_invalid_block_domain(k, budget):
    with pytest.raises(ValueError):
        ref.block_law(lambda h: [1, 0], lambda h: [1, 0], (), k, budget, 1)
