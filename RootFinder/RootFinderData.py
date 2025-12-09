from typing import Dict, Any


class RootFinderData:
    """Data Transfer Object (DTO) for root-finding problems"""

    def __init__(self, equation: str, method: str, precision: int, params: Dict[str, Any]):
        self.equation = equation
        self.method = method
        self.precision = precision
        self.params = params