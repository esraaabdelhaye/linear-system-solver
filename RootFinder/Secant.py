from typing import Dict, Any

from numpy.f2py.auxfuncs import throw_error

from RootFinder import AbstractRootFinder


class Secant(AbstractRootFinder):
    """Secant Method Implementation"""

    def solve(self) -> Dict[str, Any]:
        initial_guess = float(self.params.get("initial_guess", 1))
        second_guess = float(self.params.get("second_guess", 2))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))
        tol = 10 ** -(self.precision + 1)



        xi_minus1 = self.round_sig_fig(initial_guess)
        xi = self.round_sig_fig(second_guess)
        fxi_minus1 = self.round_sig_fig(self.evaluate(xi_minus1))
        fxi = self.round_sig_fig(self.evaluate(xi))
        denom = fxi_minus1 - fxi

        for i in range(max_iter):
            if abs(denom) < tol:
                raise ValueError(f"Division by zero detected (secant denominator too small) at iteration {i+1}")
            xi_plus1 = self.round_sig_fig(xi - fxi * (xi_minus1 - xi) / denom)

            rel_error = self.round_sig_fig(abs((xi_plus1 - xi) / max(abs(xi_plus1), 1e-12)))

            if rel_error > 1e12:
                raise ValueError(f"Divergence detected at iteration {i + 1}. Relative error too large.")

            self.add_step({
                "iteration": i + 1,
                "x\u1d62\u208B\u2081": xi_minus1,
                "x\u1d62": xi,
                "f(x\u1d62\u208B\u2081)": fxi_minus1,
                "f(x\u1d62)": fxi,
                "x\u1d62₊\u2081": xi_plus1,
                "rel_error": rel_error
            })

            if rel_error < eps:
                return {
                    "success": True,
                    "root": xi_plus1,
                    "iterations": i+1,
                    "rel_error": rel_error,
                    "correct_sig_figs": self._count_correct_sig_figs(xi_plus1, xi),
                    "steps": self.steps
                }

            xi_minus1 = xi
            xi = xi_plus1
            fxi_minus1 = self.round_sig_fig(self.evaluate(xi_minus1))
            fxi = self.round_sig_fig(self.evaluate(xi))
            denom = fxi_minus1 - fxi

        raise ValueError(
            f"Secant method failed to converge after {max_iter} iterations.\n"
            f"Last approximation: {xi_plus1}"
        )

