"""Shared, lightweight image-export options for the GUI and worker."""

from dataclasses import asdict, dataclass


def scaled_image_size(size: tuple[int, int], scale_percent: int, max_dimension: int):
    """Scale proportionally, then optionally cap the longest edge (never pad)."""
    factor = scale_percent / 100
    if max_dimension:
        factor = min(factor, max_dimension / max(size))
    return tuple(max(1, round(dimension * factor)) for dimension in size)


@dataclass(frozen=True)
class ImageExportSettings:
    dpi: int = 192
    scale_percent: int = 100
    max_dimension: int = 0

    def __post_init__(self):
        for value, lower, upper, label in (
            (self.dpi, 72, 600, "Resolution (DPI)"),
            (self.scale_percent, 10, 400, "Scale (%)"),
            (self.max_dimension, 0, 20000, "Maximum longest edge (pixels)"),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{label} must be a whole number from {lower} to {upper}.")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Image export settings must be an object.")
        return cls(**{key: values[key] for key in asdict(cls()) if key in values})

    def to_dict(self) -> dict:
        return asdict(self)

    def converter_config(self) -> dict:
        return {
            "image_export_dpi": self.dpi,
            "image_export_scale_percent": self.scale_percent,
            "image_export_max_dimension": self.max_dimension,
        }

    def summary(self) -> str:
        limit = f" · longest edge ≤ {self.max_dimension:,} px" if self.max_dimension else ""
        return f"Photos: {self.dpi} DPI · {self.scale_percent}% scale{limit}"
