from typing import Dict, Any

from .AbstractRootFinder import AbstractRootFinder


class FixedPoint(AbstractRootFinder):
    """Fixed Point Iteration Method Implementation"""

    def solve(self) -> Dict[str, Any]:
        x0 = float(self.params.get("initial_guess", 1.0))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))

        x = x0

        for iteration in range(max_iter):
            try:
                gx = self.evaluate(x)
                rel_error = abs((gx - x) / (gx if gx != 0 else 1e-10))

                self.add_step({
                    "iteration": iteration + 1,
                    "x": self.round_sig_fig(x),
                    "g(x)": self.round_sig_fig(gx),
                    "rel_error": self.round_sig_fig(rel_error)
                })

                if rel_error < eps or abs(gx - x) < eps:
                    return {
                        "success": True,
                        "root": self.round_sig_fig(gx),
                        "iterations": iteration + 1,
                        "rel_error": self.round_sig_fig(rel_error),
                        "correct_sig_figs": self._count_correct_sig_figs(gx, x),
                        "steps": self.steps
                    }

                if abs(gx) > 1e10:
                    raise ValueError(f"Method diverging: |g(x)| = {abs(gx)} is too large")

                x = gx

            except ValueError as e:
                raise ValueError(f"Fixed point method diverged at iteration {iteration + 1}: {e}")

        raise ValueError(
            f"Fixed Point method failed to converge after {max_iter} iterations.\n"
            f"Last approximation: {x}"
        )

