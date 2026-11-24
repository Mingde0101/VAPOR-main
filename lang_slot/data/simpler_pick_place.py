"""Reader specialization for the filtered SIMPLER-like pick/place subset.

The exported subset is still Bridge RLDS/TFRecord data, so this module reuses
the generic Bridge policy reader and only makes the export manifest mandatory.
This keeps future task-specific readers easy to locate without duplicating the
image/action decoding implementation.
"""

from __future__ import annotations

from typing import Any, Optional

from .bridge_rlds import PolicyWindowDataset
from ..paths import data_path


DEFAULT_SIMPLER_PICK_PLACE_TFDS_DIR = str(
    data_path("Bridge", "simpler_pick_place_subset", "1.0.0")
)


class SimplerPickPlacePolicyWindowDataset(PolicyWindowDataset):
    """Emit policy windows plus the subset's filter labels and provenance.

    The manifest row is exposed as ``PolicyWindowSample.trajectory_metadata``.
    A missing row is treated as an error because it would break the mapping
    between the exported trajectory and its task category.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_SIMPLER_PICK_PLACE_TFDS_DIR,
        *,
        metadata_manifest_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            metadata_manifest_path=metadata_manifest_path,
            require_metadata_manifest=True,
            **kwargs,
        )
