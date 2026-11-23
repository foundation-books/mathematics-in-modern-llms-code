"""Behavioral checks for constrained designs and exact fixture provenance."""

import pytest
import sympy as sp

from reference import (effective_exposures, fit_fixture, lifetime_optimum,
                       loss, n, d, reciprocal_optimum, verify_all)
from expansion_reference import evaluation_sampling, mean_statistics
from expansion_student import (EVALUATION_DOCUMENT_COUNTS,
                               EVALUATION_PAIRED_COUNTS)


def test_only_requested_grid_response_changes():
    grid, design, original, _ = fit_fixture()
    other_grid, other_design, changed, _ = fit_fixture(perturb=True)
    assert other_grid == grid and other_design == design
    differences = changed - original
    changed_rows = [i for i, value in enumerate(differences) if value != 0]
    assert [grid[i] for i in changed_rows] == [(4, 18)]
    assert differences[changed_rows[0]] == sp.Rational(1, 50)


def test_cap_changes_feasibility_and_selects_boundary():
    size, tokens = reciprocal_optimum(144, token_cap=12)
    assert size * tokens == 144 and tokens <= 12
    selected_loss = loss.subs({n: size, d: tokens})
    for alternative in (13, 16, 24):
        alternative_tokens = sp.Rational(144, alternative)
        assert alternative_tokens <= 12
        assert loss.subs({n: alternative, d: alternative_tokens}) > selected_loss
    assert reciprocal_optimum(144, token_cap=20) == (8, 18)


def test_repetition_endpoints_and_paid_count():
    for passes in (1, 3, 7):
        assert effective_exposures(6, passes, 0) == 6
        assert effective_exposures(6, passes, 1) == 6 * passes
        discounted = effective_exposures(6, passes, sp.Rational(1, 2))
        assert 6 <= discounted <= 6 * passes
        assert discounted < 12
    with pytest.raises(ValueError):
        effective_exposures(6, 0, sp.Rational(1, 2))
    with pytest.raises(ValueError):
        effective_exposures(6, 3, sp.Rational(3, 2))


def test_lifetime_designs_preserve_quality_and_improve_declared_cost():
    for weight in (0, 7, 55):
        size, tokens = lifetime_optimum(weight)
        assert size > 4
        assert sp.simplify(loss.subs({n: size, d: tokens})) == 2
        assert sp.simplify(size * (tokens + weight)) <= 8 * (18 + weight)
    with pytest.raises(ValueError):
        lifetime_optimum(-1)


def test_all_exact_assessment_anchors_and_positive_regret():
    _, size, tokens, regret = verify_all()
    assert sp.simplify(size * tokens) == 144
    assert regret > 0
    assert sp.N(regret, 25) < sp.Rational(1, 10000)


def test_resource_domain_validation():
    with pytest.raises(ValueError):
        reciprocal_optimum(0)
    with pytest.raises(ValueError):
        reciprocal_optimum(144, token_cap=0)


def test_evaluation_units_preserve_mean_but_change_uncertainty():
    assert tuple(map(sum, zip(*EVALUATION_DOCUMENT_COUNTS))) == EVALUATION_PAIRED_COUNTS
    result = evaluation_sampling()
    assert (result["item_n"], result["document_n"]) == (20, 4)
    assert result["item_mean"] == result["document_mean"] == sp.Rational(1, 5)
    assert result["document_means"] == (0, sp.Rational(2, 5),
                                         sp.Rational(3, 5), -sp.Rational(1, 5))
    assert result["item_variance"] == sp.Rational(36, 95)
    assert result["document_variance"] == sp.Rational(2, 15)
    assert result["item_se2"] == sp.Rational(9, 475)
    assert result["document_se2"] == sp.Rational(1, 30)


def test_fixed_sample_item_test_and_prespecified_family():
    result = evaluation_sampling()
    assert result["iid_item_exact_pvalue"] == sp.Rational(37, 128)
    assert "document_pvalue" not in result
    assert result["family_size"] == 10
    assert result["local_threshold"] == sp.Rational(1, 200)


def test_estimated_mean_variance_uses_independent_unit_count():
    assert mean_statistics([0, 1]) == (sp.Rational(1, 2),
                                      sp.Rational(1, 2), sp.Rational(1, 4))
    assert mean_statistics([sp.Rational(1, 5)] * 4) == (sp.Rational(1, 5), 0, 0)
    with pytest.raises(ValueError):
        mean_statistics([1])
