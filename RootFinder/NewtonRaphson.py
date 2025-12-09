from typing import Dict, Any
import math
from .AbstractRootFinder import AbstractRootFinder


class NewtonRaphson(AbstractRootFinder):

    def solve(self) -> Dict[str, Any]:
        x0 = float(self.params.get("initial_guess", 1.0))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))

        x_prev = self.round_sig_fig(x0)
        fx = self.round_sig_fig(self.evaluate(x_prev))

        # Check if initial guess is at a vertical asymptote
        if not self._is_finite(fx):
            raise ValueError(
                f"Function value is infinite at initial guess x = {x_prev}.\n"
                f"f({x_prev}) = {fx}\n"
                f"Please choose a different initial guess away from vertical asymptotes."
            )

        for iteration in range(max_iter):
            # Compute derivative
            dfx = self.round_sig_fig(self.evaluate_first_derivative(x_prev))

            if abs(dfx) < 1e-12:
                raise ValueError(
                    f"Derivative too close to zero at x = {x_prev}.\n"
                    f"f'({x_prev}) = {dfx}\n"
                    f"Cannot continue with Newton-Raphson method."
                )


            x_new = self.round_sig_fig(x_prev - (fx / dfx))
            fx_new = self.round_sig_fig(self.evaluate(x_new))

            # Check if we've hit a vertical asymptote
            if not self._is_finite(fx_new):
                raise ValueError(
                    f"Function value became infinite at x = {x_new}.\n"
                    f"f({x_new}) = {fx_new}\n"
                    f"The iteration encountered a vertical asymptote."
                )

            rel_error = self.round_sig_fig(abs((x_new - x_prev) / (x_new if x_new != 0 else 1e-10)))

            self.add_step({
                "iteration": iteration + 1,
                "x": x_prev,
                "f(x)": fx,
                "f'(x)": dfx,
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

        raise ValueError(
            f"Newton-Raphson method failed to converge after {max_iter} iterations.\n"
            f"Last approximation: {x_prev}"
        )

    def _is_finite(self, value: float) -> bool:
        return math.isfinite(value)