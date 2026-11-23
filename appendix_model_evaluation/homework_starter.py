"""Exact inputs for one three-part evaluation investigation; no answers."""

import platform

import sympy as sp


# Counts are (both correct, A only, B only, neither correct).
PAIRED_COUNTS = (8, 6, 2, 4)
DOCUMENT_COUNTS = ((5, 0, 0, 0), (3, 2, 0, 0), (0, 4, 1, 0), (0, 0, 1, 4))
FACTORIAL = {(0, 0): sp.Rational(3, 5), (1, 0): sp.Rational(7, 10),
             (0, 1): sp.Rational(13, 20), (1, 1): sp.Rational(4, 5)}
STRATA = {"A": (sp.Rational(9, 10), sp.Rational(2, 5)),
          "B": (sp.Rational(4, 5), sp.Rational(3, 5))}
# These configurations are separate from the systems in the other examples.
POINTS = {"A": (sp.Rational(4, 5), 4), "B": (sp.Rational(3, 4), 2),
          "C": (sp.Rational(7, 10), 3), "D": (sp.Rational(17, 20), 8),
          "E": (sp.Rational(77, 100), 3)}


def main():
    print(f"Python {platform.python_version()}; SymPy {sp.__version__}")
    print("Part I: construct pairs; derive means, covariance, SE and exact test;")
    print("        calculate document means and multiplicity; check boundary cases.")
    print("Part II: derive conditional/main effects, interaction and task mixtures.")
    print("Part III: derive dominance, all utility regions/ties and the cost-cap choice.")
    # Add your derived expressions and exact calculations for each part here.
    # Explain the result of each part; finish with the constrained choice.


if __name__ == "__main__":
    main()
