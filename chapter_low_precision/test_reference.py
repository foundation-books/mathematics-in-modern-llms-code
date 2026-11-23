"""Behavioral checks with an independent Fraction/round encoding oracle."""
from fractions import Fraction
from pathlib import Path
import re
import pytest
import sympy as sp

from fixture_checks import UniformQuantizer, round_even
from reference import (QUANT, calibration_comparison, clipped_ste_mask, decode, max_quantizer, part_a,
                       part_b, part_c, part_d, part_e, part_f, scale_surrogate,
                       signed_width)


@pytest.mark.parametrize("scale,lower,upper,zero", [
    (Fraction(1, 4), -7, 7, 0), (Fraction(1, 2), 0, 7, 3),
    (Fraction(2, 3), -8, 7, -2)])
def test_quantizer_against_independent_exact_oracle(scale, lower, upper, zero):
    quant = UniformQuantizer(sp.Rational(scale.numerator, scale.denominator), lower, upper, zero)
    for numerator in range(-72, 73):
        value = Fraction(numerator, 16)
        expected_code = min(upper, max(lower, round(value/scale) + zero))
        exact_value = sp.Rational(value.numerator, value.denominator)
        assert quant.encode(exact_value) == expected_code
        expected_value = scale*(expected_code-zero)
        assert quant(exact_value) == sp.Rational(expected_value.numerator, expected_value.denominator)
        assert decode(expected_code, quant) == quant(exact_value)


@pytest.mark.parametrize("call", [
    lambda: round_even(0.25), lambda: UniformQuantizer(0),
    lambda: UniformQuantizer(-1), lambda: UniformQuantizer(1, 7, 7),
    lambda: UniformQuantizer(1, -7, 7, 8),
    lambda: UniformQuantizer(1, -7.0, 7),
    lambda: decode(sp.Rational(1, 2)), lambda: decode(8)])
def test_invalid_representation_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_zero_tensor_scale_policy():
    tensor = sp.zeros(3, 2)
    quant = max_quantizer(tensor)
    assert quant.delta == 1 and quant.matrix(tensor) == tensor


def test_clipped_ste_uses_strict_interval_with_zero_boundaries():
    values = sp.Matrix([sp.Rational(-7, 4), -1, 0, 1, sp.Rational(7, 4), 2])
    assert clipped_ste_mask(values) == sp.Matrix([0, 1, 1, 1, 0, 0])


def test_scale_backward_rule_saturation_and_nonzero_zero_rejection():
    assert scale_surrogate(2) == 7 and scale_surrogate(-2) == -7
    assert scale_surrogate(sp.Rational(1, 3)) == sp.Rational(-1, 3)
    with pytest.raises(ValueError):
        scale_surrogate(1, UniformQuantizer(1, 0, 7, 3))


def test_exact_reference_tasks():
    for function in (part_a, part_b, part_c, part_d, part_e, part_f):
        assert function()


@pytest.mark.parametrize("a,b", [
    (Fraction(1, 100), Fraction(10)), (Fraction(1), Fraction(5)),
    (Fraction(1), Fraction(10)), (Fraction(1), Fraction(20))])
def test_stipulated_reconstructions_use_two_column_normalization(a, b):
    result = calibration_comparison(sp.Rational(a.numerator, a.denominator),
                                    sp.Rational(b.numerator, b.denominator))
    assert result["count"] == 2
    rows = result["candidates"]
    # Independent scalar Fraction arithmetic checks both strict orderings and tie.
    expected = (b*b/4, 25*a*a)
    assert [row["weight_norm"] for row in rows] == [1, 10]
    for row, loss in zip(rows, expected):
        assert row["loss"] == row["trace_loss"] == sp.Rational(loss.numerator, loss.denominator)
    assert rows[0]["residual"] == sp.diag(0, -1)
    assert rows[1]["residual"] == sp.diag(-10, 0)
    assert (rows[0]["loss"] > rows[1]["loss"]) == (b > 10*a)


def test_symbolic_calibration_comparison_and_original_homework_are_independent():
    a, b = sp.symbols("a b", positive=True)
    rows = calibration_comparison(a, b)["candidates"]
    assert [row["loss"] for row in rows] == [b**2/4, 25*a**2]
    assert sp.simplify(rows[0]["loss"]-rows[1]["loss"]-(b-10*a)*(b+10*a)/4) == 0
    assert part_c()["moment"] == sp.Matrix([
        [sp.Rational(13, 27), sp.Rational(5, 27)],
        [sp.Rational(5, 27), sp.Rational(61, 108)]])
    assert part_c()["reconstruction_loss"] == sp.Rational(73, 31104)


def test_relative_error_counterexample_is_unbounded():
    eps = sp.symbols("eps", positive=True)
    # For 0<eps<1/8 the two entries lie on opposite sides of a rounding jump.
    relative_error = (sp.Rational(1, 4)-2*eps)/(2*eps)
    assert sp.limit(relative_error, eps, 0, dir="+") == sp.oo
    for value in (sp.Rational(1, 64), sp.Rational(1, 128)):
        x = sp.Matrix([sp.Rational(1, 8)-value, sp.Rational(1, 8)+value])
        assert (sp.Matrix([[1, -1]])*QUANT.matrix(x))[0] == sp.Rational(-1, 4)


def test_nonzero_zero_points_centered_dot_product():
    qw, qx = sp.Matrix([1, 3, 7]), sp.Matrix([2, 5, 6])
    zw, zx = 3, 4
    direct = sum((w-zw)*(x-zx) for w, x in zip(qw, qx))
    expanded = qw.dot(qx)-zx*sum(qw)-zw*sum(qx)+3*zw*zx
    assert direct == expanded == 12


def test_conservative_signed_width_is_minimal_for_bound():
    assert signed_width(294) == 10
    for bound in (0, 1, 7, 8, 255, 256, 294, 511, 512):
        width = signed_width(bound)
        assert bound <= 2**(width-1)-1
        assert width == 1 or bound > 2**(width-2)-1


def test_student_starter_imports_no_answers():
    starter = (Path(__file__).parent/'homework_student.py').read_text()
    assert 'from reference' not in starter and 'fixture_checks' not in starter


def test_printed_specification_keeps_inputs_and_rounding_policy():
    root = Path(__file__).resolve().parents[2]
    source = root/'book/chapter_low_precision/sections/10_homework.tex'
    if not source.is_file():
        pytest.skip("Manuscript parity check requires the source repository")
    manuscript = source.read_text()
    for required in ("sec:precision-contract", "nearest-even", "scale $\\Delta=1/4$",
                     "integer bounds $[-7,7]$", "zero point 0", "target $(0,1)^\\top$"):
        assert required in manuscript
    assert "\\begin{lstlisting}" not in manuscript
