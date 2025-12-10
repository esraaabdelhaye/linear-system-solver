from typing import Dict, Any

from numpy.f2py.auxfuncs import throw_error

from RootFinder import AbstractRootFinder


class ModifiedSecant(AbstractRootFinder):
    """Secant Method Implementation"""

    def solve(self) -> Dict[str, Any]:
        initial_guess = float(self.params.get("initial_guess", 1))
        delta = float(self.params.get("delta", 0.01))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))
        tol = 10 ** -(self.precision + 1)
        delta = self.round_sig_fig(delta)



        xi = self.round_sig_fig(initial_guess)
        xi_plus_delta = self.round_sig_fig(xi + delta * xi)
        fxi = self.round_sig_fig(self.evaluate(xi))
        fxi_plus_delta = self.round_sig_fig(self.evaluate(xi_plus_delta))
        denom = fxi_plus_delta - fxi

        for i in range(max_iter):
            if abs(denom) < tol:
                raise ValueError(f"Division by zero detected (modified secant denominator too small) at iteration {i+1}")
            xi_plus1 = self.round_sig_fig(xi - fxi * (delta * xi) / denom)

            rel_error = self.round_sig_fig(abs((xi_plus1 - xi) / max(abs(xi_plus1), 1e-12)))

            if rel_error > 1e12:
                raise ValueError(f"Divergence detected at iteration {i + 1}. Relative error too large.")

            self.add_step({
                "iteration": i + 1,
                "x\u1d62": xi,
                "x\u1d62 + ẟx\u1d62": xi_plus_delta,
                "f(x\u1d62)": fxi,
                "f(x\u1d62 + ẟx\u1d62)": fxi_plus_delta,
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

            xi = xi_plus1
            xi_plus_delta = self.round_sig_fig(xi + delta * xi)
            fxi = self.round_sig_fig(self.evaluate(xi))
            fxi_plus_delta = self.round_sig_fig(self.evaluate(xi_plus_delta))
            denom = fxi_plus_delta - fxi

        raise ValueError(
            f"Secant method failed to converge after {max_iter} iterations.\n"
            f"Last approximation: {xi_plus1}"
        )

