from typing import Dict, Any
import math
from .AbstractRootFinder import AbstractRootFinder


class Modified1NewtonRaphson(AbstractRootFinder):
    """Modified Newton Raphson Method when multiplicity m is KNOWN"""

    def solve(self) -> Dict[str, Any]:
        x0 = float(self.params.get("initial_guess", 1.0))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))
        m = float(self.params.get("multiplicity", None))

        if m is None:
            raise ValueError("Multiplicity 'm' must be provided for Modified Newton Raphson Method 1")

        x_prev = self.round_sig_fig(x0)
        fx = self.round_sig_fig(self.evaluate(x_prev))

        if not self._is_finite(fx):
            raise ValueError(
                f"Function value is infinite at initial guess x = {x_prev}.\n"
                f"f({x_prev}) = {fx}\n"
                f"Please choose a different initial guess away from vertical asymptotes."
            )


        for iteration in range(max_iter):

            dfx = self.round_sig_fig(self.evaluate_first_derivative(x_prev))

            if abs(dfx) < 1e-12:
                raise ValueError(f"Derivative too small at x = {x_prev}")

            # --- Modified NR formula x_{i+1} = x_i - m * f(x)/f'(x)
            step = self.round_sig_fig(fx / dfx)
            step = self.round_sig_fig(m * step)
            x_new = self.round_sig_fig(x_prev - step)

            fx_new = self.round_sig_fig(self.evaluate(x_new))
            rel_error = self.round_sig_fig(abs((x_new - x_prev) / (x_new if x_new != 0 else 1e-10)))

            self.add_step({
                "iteration": iteration + 1,
                "x": x_prev,
                "f(x)": fx,
                "f'(x)": dfx,
                "m": m,
                "x_new": x_new,
                "rel_error": rel_error
            })

            if rel_error < eps:
                return {
                    "success": True,
                    "root": x_new,
                    "iterations": iteration + 1,
                    "rel_error": rel_error,
                    "correct_sig_figs": self._count_correct_sig_figs(x_new, x_prev),
                    "steps": self.steps
                }

            x_prev = x_new
            fx = fx_new

        raise ValueError(f"Modified Newton–Raphson (Method 1) failed to converge after {max_iter} iterations.")


    def _is_finite(self, value: float) -> bool:
        return math.isfinite(value)