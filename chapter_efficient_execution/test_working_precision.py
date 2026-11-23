"""Check rounding behavior, independence and input-domain failures."""
from fractions import Fraction
import mpmath as mp
import pytest
from working_precision import WorkingArithmetic, high_precision_weighted_reference


def test_working_precision_changes_intermediate_arithmetic():
    original_digits = mp.mp.dps
    low, high = WorkingArithmetic(8), WorkingArithmetic(12)
    assert low.sub(low.add("10000000000", 1), "10000000000") == 0
    assert high.sub(high.add("10000000000", 1), "10000000000") == 1
    # A final display with eight digits could not reproduce this distinction.
    assert mp.mp.dps == original_digits  # The helper never changes the caller's context.


def test_nontrivial_reference_retains_exact_signed_cancellation():
    rows = ((10**12, 1), (-10**12, 1), (1, -2))
    result = high_precision_weighted_reference((1, 1, 1), rows)
    high = WorkingArithmetic(70)
    assert result == (high.number(Fraction(1, 3)), high.number(0))
    low = WorkingArithmetic(8)
    assert low.add(low.add(10**12, 1), -10**12) == 0
    assert result[0] > 0  # The independent reference did not lose the unit.


def test_operations_round_separately_and_reject_nonfinite_inputs():
    arithmetic = WorkingArithmetic(12)
    assert arithmetic.div(arithmetic.mul(3, 4), 2) == 6
    assert arithmetic.exp(arithmetic.log(1)) == 1
    for bad in (float("inf"), "inf", "nan", 0.1):
        with pytest.raises(ValueError):
            arithmetic.number(bad)
    with pytest.raises(ValueError):
        arithmetic.log(0)
    with pytest.raises(ZeroDivisionError):
        arithmetic.div(1, 0)
    with pytest.raises(ValueError):
        high_precision_weighted_reference((1, 2), ((1,), (2, 3)))


def test_fraction_conversion_rounds_exact_quotient_only_once():
    arithmetic = WorkingArithmetic(2)  # 10 binary significant bits.
    exact = Fraction(1010, 1027)
    expected = arithmetic.context.mpf(1007) / 1024
    separately_rounded = arithmetic.context.mpf(503) / 512
    assert arithmetic.context.prec == 10
    assert arithmetic.number(exact) == expected
    assert arithmetic.number(exact) != separately_rounded
    assert high_precision_weighted_reference((1,), ((exact,),), digits=2) == (expected,)
    # Independent exact distances verify the nearest representable neighbor.
    assert abs(exact - Fraction(1007, 1024)) < abs(exact - Fraction(1006, 1024))
    assert abs(exact - Fraction(1007, 1024)) < abs(exact - Fraction(1008, 1024))
