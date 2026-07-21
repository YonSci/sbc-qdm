"""Registry of bias-correction methods compared against QDM.

Every entry exposes a `train_fn(ref, hist, cfg) -> trained_state` /
`apply_fn(sim, trained, cfg) -> xr.DataArray` pair with the same signature as
qdm.py's train_qdm/apply_qdm, so cli.py can dispatch on method name generically.
`trained_state`'s shape differs per method (see each module's docstring) --
callers that need to serialize it (see cli.py's _trained_to_dataset-style
helpers) branch on the method name to know which shape to expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import xarray as xr

from sbc_qdm.methods.delta_change import apply_delta_change, train_delta_change
from sbc_qdm.methods.detrended_quantile_mapping import apply_detrended_quantile_mapping, train_detrended_quantile_mapping
from sbc_qdm.methods.empirical_quantile_mapping import apply_empirical_quantile_mapping, train_empirical_quantile_mapping
from sbc_qdm.methods.linear_scaling import apply_linear_scaling, train_linear_scaling
from sbc_qdm.methods.power_transformation import apply_power_transformation, train_power_transformation
from sbc_qdm.methods.variance_scaling import apply_variance_scaling, train_variance_scaling
from sbc_qdm.qdm import apply_qdm, train_qdm


@dataclass(frozen=True)
class MethodSpec:
    train_fn: Callable[[xr.DataArray, xr.DataArray, dict], Any]
    apply_fn: Callable[[xr.DataArray, Any, dict], xr.DataArray]
    display_name: str


METHODS: dict[str, MethodSpec] = {
    "qdm": MethodSpec(train_qdm, apply_qdm, "Quantile Delta Mapping"),
    "linear_scaling": MethodSpec(train_linear_scaling, apply_linear_scaling, "Linear Scaling"),
    "delta_change": MethodSpec(train_delta_change, apply_delta_change, "Delta Change"),
    "variance_scaling": MethodSpec(train_variance_scaling, apply_variance_scaling, "Variance Scaling"),
    "power_transformation": MethodSpec(train_power_transformation, apply_power_transformation, "Power Transformation"),
    "empirical_quantile_mapping": MethodSpec(train_empirical_quantile_mapping, apply_empirical_quantile_mapping, "Empirical Quantile Mapping"),
    "detrended_quantile_mapping": MethodSpec(train_detrended_quantile_mapping, apply_detrended_quantile_mapping, "Detrended Quantile Mapping"),
}
