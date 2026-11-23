"""Primitive arithmetic for Part D, without an attention implementation.

Use integer, decimal-string or Fraction inputs, never pre-rounded Python floats.
Each conversion and primitive operation rounds to nearest at the cloned mpmath
context's binary precision. Intermediate products and sums round separately.
The exact-weight reference does not reuse the working-precision reduction.
"""
from fractions import Fraction
import mpmath as mp


class WorkingArithmetic:
    def __init__(self, digits=8):
        if type(digits) is not int or digits < 2:
            raise ValueError("Use at least two decimal working digits.")
        self.context = mp.mp.clone()
        self.context.dps = digits

    def number(self, value):
        if isinstance(value, float):
            raise ValueError("Construct inputs from integers, strings or Fractions.")
        if isinstance(value, Fraction):
            # fdiv accepts exact integer operands. Converting each to mpf first
            # can round the operands before the quotient and change its result.
            result = self.context.fdiv(value.numerator, value.denominator)
        else:
            result = self.context.mpf(value)
        if not self.context.isfinite(result):
            raise ValueError("Finite real inputs and results are required.")
        return result

    def add(self, left, right):
        return self.number(self.context.fadd(self.number(left), self.number(right)))

    def sub(self, left, right):
        return self.number(self.context.fsub(self.number(left), self.number(right)))

    def mul(self, left, right):
        return self.number(self.context.fmul(self.number(left), self.number(right)))

    def div(self, left, right):
        return self.number(self.context.fdiv(self.number(left), self.number(right)))

    def log(self, value):
        value = self.number(value)
        if value <= 0:
            raise ValueError("Logarithms require positive real inputs.")
        return self.number(self.context.log(value))

    def exp(self, value):
        return self.number(self.context.exp(self.number(value)))


def high_precision_weighted_reference(weights, rows, digits=70):
    """Exact rational dot products/divisor, then one high-precision conversion.

    For the printed data exp(log(integer weight)) is that exact weight; a
    common score shift cancels. Supply integer, string or Fraction operands.
    """
    if any(isinstance(x, float) for x in (*weights, *(x for row in rows for x in row))):
        raise ValueError("The reference requires exact inputs.")
    weights = tuple(map(Fraction, weights))
    rows = tuple(tuple(map(Fraction, row)) for row in rows)
    if (not weights or len(weights) != len(rows) or not rows[0]
            or any(len(row) != len(rows[0]) for row in rows)
            or any(w <= 0 for w in weights)):
        raise ValueError("Use positive weights and equally wide nonempty rows.")
    total = sum(weights)
    exact = tuple(sum(w * row[j] for w, row in zip(weights, rows)) / total
                  for j in range(len(rows[0])))
    arithmetic = WorkingArithmetic(digits)
    return tuple(arithmetic.number(x) for x in exact)


if __name__ == "__main__":
    for digits in (8, 12):
        arithmetic = WorkingArithmetic(digits)
        result = arithmetic.add("10000000000", 1)
        print(digits, arithmetic.context.prec,
              arithmetic.context.nstr(result, 12))
