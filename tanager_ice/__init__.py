"""tanager_ice: STAC ingest -> spectral retrievals -> conformal uncertainty ->
degradation, for the Tanager snow/ice hyperspectral characterisation project."""
from . import spectral, separability, uncertainty
try:
    from . import io  # noqa: F401  (optional heavy deps)
except Exception:  # pragma: no cover
    io = None
__version__ = "0.1.0"
__all__ = ["spectral", "separability", "uncertainty", "io"]
