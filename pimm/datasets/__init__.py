# pimm/datasets/__init__.py — de-fork re-export / re-register shim.
#
# The data layer now lives in pimm-data (installed editable as the
# `libs/pimm-data` submodule). pimm keeps only: MultiDatasetDataloader (DDP),
# the model/hook/loss registries, hooks/evaluators, and this shim.
#
# Config `type=` strings resolve through pimm-data's DATASETS / TRANSFORMS,
# re-exported here as `pimm.datasets.DATASETS` / `pimm.datasets.TRANSFORMS`.
# pimm-data owns the generic/IO transforms; the pretraining-recipe transforms
# (multi-view, MAE masking, anchors, instance targets) live in pimm and register
# INTO the shared pimm_data.TRANSFORMS via the eager `transforms` import below.
# Importing pimm_data registers every data-layer dataset/transform via import
# side-effects; the eager import then layers pimm's SSL transforms on top.

from pimm_data import (
    DATASETS, TRANSFORMS, Compose, build_dataset,
    DefaultDataset, ConcatDataset,
    PILArNetH5Dataset, JAXTPCDataset, LUCiDDataset,
    MultiModalEventDataset,
)
from pimm_data.collate import collate_fn, point_collate_fn, inseg_collate_fn

# Register pimm-owned SSL/pretraining transforms into the shared
# pimm_data.TRANSFORMS. Eager (not lazy) so any config referencing e.g.
# type="MultiViewGenerator" resolves before the first Compose build. Must come
# after the `from pimm_data import ... TRANSFORMS` above (registry must exist).
from . import transforms as _pimm_transforms  # noqa: F401

# The SSL config's type="LUCiDEventSSLDataset" was migrated to
# type="MultiModalEventDataset" directly, so no alias re-registration is needed.

# KEEP-IN-PIMM: the DDP dataloader (imports pimm.utils.comm / env).
from .dataloader import MultiDatasetDataloader

__all__ = [
    "DATASETS", "TRANSFORMS", "Compose", "build_dataset",
    "collate_fn", "point_collate_fn", "inseg_collate_fn",
    "DefaultDataset", "ConcatDataset",
    "PILArNetH5Dataset", "JAXTPCDataset", "LUCiDDataset",
    "MultiModalEventDataset",
    "MultiDatasetDataloader",
]
