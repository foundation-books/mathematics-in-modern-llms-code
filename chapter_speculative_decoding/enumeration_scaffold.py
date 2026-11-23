"""Bookkeeping only: supply the conditional laws and branching decisions yourself."""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Branch:
    history: tuple
    mass: Fraction


def extend(branch, token, conditional_probability):
    """Store a conditional separately; multiply it by incoming branch mass."""
    if isinstance(conditional_probability, float):
        raise ValueError("Use exact probabilities.")
    conditional = Fraction(conditional_probability)
    if not 0 <= conditional <= 1 or not 0 <= branch.mass <= 1:
        raise ValueError("Probability and incoming mass must lie in [0,1].")
    child = Branch(branch.history + (token,), branch.mass * conditional)
    return {"parent_mass": branch.mass, "conditional": conditional, "child": child}


if __name__ == "__main__":
    # This arithmetic row specifies no target, proposal or correction law.
    row = extend(Branch((), Fraction(2, 3)), "event", Fraction(1, 4))
    assert row["child"].mass == Fraction(1, 6)
    print(row)
    # Start an actual enumeration at Branch((), Fraction(1)). For each parent,
    # build the appropriate conditional row, check that it normalizes, and call
    # extend for its events. Preserve EOS and budget-failure branches explicitly.
