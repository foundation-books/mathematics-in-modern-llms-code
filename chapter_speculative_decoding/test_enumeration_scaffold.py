"""Scaffold bookkeeping must retain incoming mass and all stopped events."""
from fractions import Fraction as F
import pytest
from enumeration_scaffold import Branch, extend


def test_conditional_is_not_the_unconditional_branch_mass():
    parent = Branch(("prefix",), F(2, 3))
    rows = [extend(parent, "EOS", F(1, 4)), extend(parent, "budget failure", F(3, 4))]
    assert [row["child"].mass for row in rows] == [F(1, 6), F(1, 2)]
    assert sum(row["child"].mass for row in rows) == parent.mass
    assert rows[0]["conditional"] != rows[0]["child"].mass
    assert parent.history == ("prefix",)


@pytest.mark.parametrize("bad", [F(-1, 2), F(3, 2), 0.25])
def test_invalid_or_inexact_probability_rejected(bad):
    with pytest.raises(ValueError):
        extend(Branch((), F(1)), "event", bad)
