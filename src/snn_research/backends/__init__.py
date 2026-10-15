"""Hardware boundary: model + rule export, backend interface, constraint checks.

No hardware-specific backend is implemented; hardware capabilities are data.
"""

from .export import FORMAT, export_network, save_export
from .hardware import HardwareBackend, HardwareConstraints, Measurement, check_constraints, format_findings
