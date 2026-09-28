"""CurvLearn -- tunable-curvature representation spaces for language models.

Design A (this release): does a free curvature parameter actually move, or is flat an
attractor? Sweep the initial curvature and the embedding scale; measure where kappa escapes.
"""
from . import geometry, geometry_ref, data, model, train  # noqa: F401

__version__ = "0.1.0"
