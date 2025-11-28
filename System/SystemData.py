from typing import List, Tuple, Dict, Any, Optional
# --- 1. DATA TRANSFER OBJECT (DTO) ---

class SystemData:
    """
    Data Transfer Object (DTO) for passing system configuration and data
    from the GUI layer to the Solver layer (clean separation of concerns).
    """

    def __init__(self, A: List[List[float]], b: List[float], method: str,
                 precision: int, params: Dict[str, Any]):
        self.A = A  # Coefficient Matrix (2D list of floats)
        self.b = b  # Constant Vector (1D list of floats)
        self.method = method  # Solving method (e.g., "Gauss Elimination")
        self.precision = precision  # Number of significant figures (Specification 4)
        self.params = params  # Method-specific parameters (e.g., initial guess, boolean Jacobi)
        self.N = len(A)  # Size of the system (Number of Variables/Equations)
        self.single_step = False
        self.params["use_scaling"] = False

