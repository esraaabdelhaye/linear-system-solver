from typing import Dict, Any
import math
import sympy as sp
from .AbstractRootFinder import AbstractRootFinder


class Modified2NewtonRaphson(AbstractRootFinder):
    """Modified Newton–Raphson Method when multiplicity is UNKNOWN"""

    def solve(self) -> Dict[str, Any]:
        x0 = float(self.params.get("initial_guess", 1.0))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))

        # Prepare 2nd derivative
        d1 = sp.diff(self.expr, self.x_sym)
        d2 = sp.diff(d1, self.x_sym)

        x_prev = self.round_sig_fig(x0)
        fx = self.round_sig_fig(self.evaluate(x_prev))


        if not self._is_finite(fx):
            raise ValueError(
                f"Function value is infinite at initial guess x = {x_prev}.\n"
                f"f({x_prev}) = {fx}\n"
                f"Please choose a different initial guess away from vertical asymptotes."
            )

        for iteration in range(max_iter):

            f1 = self.round_sig_fig(float(d1.subs(self.x_sym, x_prev)))
            f2 = self.round_sig_fig(float(d2.subs(self.x_sym, x_prev)))

            if abs(f1) < 1e-12:
                raise ValueError(f"f'(x) too small at x = {x_prev}")

            numerator = fx * f1
            denominator = (f1 ** 2) - (fx * f2)

            if abs(denominator) < 1e-12:
                # If denominator is small, just stop with current approximation
                print(f"Warning: Denominator very small at iteration {iteration+1}, stopping.")
                return {
                    "success": True,
                    "root": x_prev,
                    "iterations": iteration+1,
                    "rel_error": rel_error,
                    "correct_sig_figs": self._count_correct_sig_figs(x_prev, x_prev - step),
                    "steps": self.steps
                }

            step = numerator / denominator
            x_new = self.round_sig_fig(x_prev - step)

            fx_new = self.round_sig_fig(self.evaluate(x_new))
            rel_error = self.round_sig_fig(abs((x_new - x_prev) / (x_new if x_new != 0 else 1e-10)))

            print(f"Iter {iteration+1}: x_prev={x_prev}, f(x)={fx}, f'={f1}, f''={f2}, "
                f"step={step}, x_new={x_new}, rel_error={rel_error}")

            self.add_step({
                "iteration": iteration + 1,
                "x": x_prev,
                "f(x)": fx,
                "f'(x)": f1,
                "f''(x)": f2,
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

        raise ValueError("Modified Newton–Raphson (Method 2) failed to converge.")


    def _is_finite(self, value: float) -> bool:
        return math.isfinite(value)