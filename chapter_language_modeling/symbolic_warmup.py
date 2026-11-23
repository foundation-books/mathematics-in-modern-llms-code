"""A first exact derivative: differentiate before substituting the logits."""
import sympy as sp


def warmup():
    z0, z1 = sp.symbols("z0 z1", real=True)
    loss = sp.log(sp.exp(z0) + sp.exp(z1)) - z0
    gradient = sp.Matrix([sp.diff(loss, z0), sp.diff(loss, z1)])
    at = gradient.subs({z0: sp.log(3), z1: 0}).applyfunc(sp.simplify)
    # Independent softmax-minus-one-hot calculation at this point.
    expected = sp.Matrix([sp.Rational(3, 4) - 1, sp.Rational(1, 4)])
    assert (at - expected).applyfunc(sp.simplify) == sp.zeros(2, 1)
    return at


if __name__ == "__main__":
    print(warmup())
