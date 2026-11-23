"""Student Chapter 13 starter; complete the five printed homework parts."""
import sympy as sp
BOS, EOS = 4, 3
CORE = (0, 1, 3)
REPLAY = (0, 1, 0, 1, 2, 3)
T1 = sp.Matrix([0, 2, 1, -1, 3])
T2 = sp.Matrix([1, 0, -1, 2, 1, -1, 3])

def suffix_bigrams(tokens):
    if not tokens:
        raise ValueError("Empty sequence")
    if any(type(x) is not int or x not in range(4)
           for x in tokens):
        raise ValueError("Integer token codes 0..3")
    if EOS in tokens[:-1]:
        raise ValueError("EOS must be last")
    return list(zip((BOS,) + tuple(tokens[:-1]),
                    tokens))

def address(key):
    a, b = key
    return (a + 2*b) % 5, (2*a + b) % 7

def read(key):
    i, j = address(key)
    return sp.Matrix([T1[i], T2[j]])

def assert_zero_matrix(M):
    assert M.applyfunc(sp.simplify) == sp.zeros(*M.shape)


# Additional fixed inputs for Parts B/E; these are not solutions.
E1 = sp.Matrix([[1, 0], [0, 1], [1, 1], [1, -1]])
E2 = sp.Matrix([[0, 1], [1, 0], [-1, 1], [2, 0]])
H_STAR = sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 6)])

if __name__ == "__main__":
    import sys
    print("Python:", sys.version.split()[0], "SymPy:", sp.__version__)
    for key in suffix_bigrams(CORE):
        print(key, address(key), read(key).T)
    # Complete Parts A-E below; submit the accompanying explanation.
