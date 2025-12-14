from typing import List, Tuple, Dict, Any, Optional
# --- 1. DATA TRANSFER OBJECT (DTO) ---

class SystemData:
    """
    Data Transfer Object (DTO) for passing system configuration and data
    from the GUI layer to the Solver layer (clean separation of concerns).
    """

    def __init__(self, A: List[List[float]], b: List[float], method: str,
                 precision: int, params: Dict[str, Any], use_scaling: bool = False):
        self.A = A  # Coefficient Matrix
        self.b = b  # Constant Vector
        self.method = method  # Solving method
        self.precision = precision  # Number of significant figures
        self.params = params  # Method-specific parameters
        self.N = len(A)  # Size of the system
        self.single_step = False  # Single Step Mode Flag
        self.use_scaling = use_scaling  # Use Scaling Mode Flag (BONUS #3)