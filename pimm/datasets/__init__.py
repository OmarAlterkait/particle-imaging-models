# pimm/datasets/__init__.py — de-fork re-export / re-register shim.
#
# The data layer now lives in pimm-data (installed editable as the
# `libs/pimm-data` submodule). pimm keeps only: MultiDatasetDataloader (DDP),
# the model/hook/loss registries, hooks/evaluators, and this shim.
#
# Config `type=` strings resolve through pimm-data's DATASETS / TRANSFORMS,
# re-exported here as `pimm.datasets.DATASETS` / `pimm.datasets.TRANSFORMS`.
# Those registries are a strict SUPERSET of the old vendored ones — pimm-data's
# transforms ⊇ pimm's (it adds ApplyToStream + RemapSegment), and the datasets
# match except the dissolved LUCiDEventSSLDataset (→ MultiModalEventDataset).
# Importing pimm_data registers every dataset/transform via import side-effects,
# so no manual re-registration is needed except the dissolved-SSL alias below.

from pimm_data import (
    DATASETS, TRANSFORMS, Compose, build_dataset,
    DefaultDataset, ConcatDataset,
    PILArNetH5Dataset, JAXTPCDataset, LUCiDDataset,
    MultiModalEventDataset,
)
from pimm_data.collate import collate_fn, point_collate_fn, inseg_collate_fn

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
