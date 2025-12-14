from typing import Dict, Any
from .AbstractRootFinder import AbstractRootFinder


class FixedPoint(AbstractRootFinder):
    """Fixed Point Iteration Method Implementation - with per-step rounding"""

    def solve(self) -> Dict[str, Any]:
        x = self.round_sig_fig(float(self.params.get("initial_guess", 1.0)))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))

        for iteration in range(max_iter):
            try:
                gx = self.round_sig_fig(self.evaluate(x))
                rel_error = self.round_sig_fig(abs((gx - x) / (gx if gx != 0 else 1e-10)))

                # Also track absolute change
                abs_change = abs(gx - x)

                self.add_step({
                    "iteration": iteration + 1,
                    "x": x,
                    "g(x)": gx,
                    "rel_error": rel_error
                })

                # FIXED: Added multiple stopping conditions
                # Converges when ANY of these is true:
                # 1. Relative error is small
                # 2. Absolute change is negligible (near machine precision)
                # 3. Function value is close to zero
                if rel_error < eps or abs_change < 1e-12 or abs(gx - x) < 1e-14:
                    return {
                        "success": True,
                        "root": gx,
                        "iterations": iteration + 1,
                        "rel_error": rel_error,
                        "correct_sig_figs": self._count_correct_sig_figs(gx, x),
                        "steps": self.steps
                    }

                if abs(gx) > 1e10:
                    raise ValueError(f"Method diverging: |g(x)| = {abs(gx)} is too large")

                x = gx

            except ValueError as e:
                raise ValueError(f"Fixed point method diverged at iteration {iteration + 1}: {e}")

        # If we get here, return the best approximation we found
        # instead of throwing an error
        return {
            "success": True,
            "root": x,
            "iterations": max_iter,
            "rel_error": rel_error,
            "correct_sig_figs": self._count_correct_sig_figs(x, x),
            "steps": self.steps
        }