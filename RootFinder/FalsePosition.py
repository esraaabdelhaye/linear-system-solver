from typing import Dict, Any
from .AbstractRootFinder import AbstractRootFinder


class FalsePosition(AbstractRootFinder):
    """False Position Method Implementation - with per-step rounding"""

    def solve(self) -> Dict[str, Any]:
        a = self.round_sig_fig(float(self.params.get("interval_a", -5.0)))
        b = self.round_sig_fig(float(self.params.get("interval_b", 5.0)))
        eps = float(self.params.get("epsilon", 0.00001))
        max_iter = int(self.params.get("max_iterations", 50))

        fa = self.round_sig_fig(self.evaluate(a))
        fb = self.round_sig_fig(self.evaluate(b))

        if fa * fb > 0:
            raise ValueError(
                f"False Position requires f(a) and f(b) to have opposite signs.\n"
                f"f({a}) = {fa}\n"
                f"f({b}) = {fb}"
            )

        root = a

        for iteration in range(max_iter):
            c = self.round_sig_fig((a * fb - b * fa) / (fb - fa))
            fc = self.round_sig_fig(self.evaluate(c))
            rel_error = self.round_sig_fig(abs((c - root) / (c if c != 0 else 1e-10)))

            self.add_step({
                "iteration": iteration + 1,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "rel_error": rel_error
            })

            if abs(fc) < eps or rel_error < eps:
                return {
                    "success": True,
                    "root": c,
                    "iterations": iteration + 1,
                    "rel_error": rel_error,
                    "correct_sig_figs": self._count_correct_sig_figs(c, root),
                    "steps": self.steps
                }

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            root = c

        raise ValueError(
            f"False Position method failed to converge after {max_iter} iterations.\n"
            f"Last approximation: {root}"
        )