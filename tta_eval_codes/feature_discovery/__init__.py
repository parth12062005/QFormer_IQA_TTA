"""
Feature Discovery Registry for AGIQA.

Each feature extractor module registers its features via FEATURE_REGISTRY.
A feature is a callable:  f(image_np_uint8_HWC, **kwargs) -> dict[str, float]
"""

FEATURE_REGISTRY = {}   # name -> callable
FEATURE_META = {}       # name -> {description, differentiable, batch_computable, family}


def register_feature(name, fn, description="", differentiable=False,
                     batch_computable=True, family="misc"):
    """Register a feature extractor function."""
    FEATURE_REGISTRY[name] = fn
    FEATURE_META[name] = {
        "description": description,
        "differentiable": differentiable,
        "batch_computable": batch_computable,
        "family": family,
    }


def register_extractor(extractor_class):
    """Register all features from an extractor class that has a `compute(img)` method
    returning dict[str, float] and class-level META dict."""
    for name, meta in extractor_class.META.items():
        def _make_fn(cls, feat_name):
            def _fn(img, **kwargs):
                result = cls().compute(img, **kwargs)
                return {feat_name: result[feat_name]}
            return _fn
        FEATURE_REGISTRY[name] = _make_fn(extractor_class, name)
        FEATURE_META[name] = meta
    return extractor_class
