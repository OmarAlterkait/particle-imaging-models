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
    compute_anchors, ANCHOR_DEFAULT_CFG,
)
from pimm_data.collate import collate_fn, point_collate_fn, inseg_collate_fn


# DISSOLVED LUCiDEventSSLDataset: the SSL config's type="LUCiDEventSSLDataset"
# resolves to the MultiModalEventDataset successor (the base + the LUCiD-SSL
# config now own the holdout / min-points / aggregation it did inline). The old
# pimm __init__ never imported lucid_event_ssl — registration was a config-side
# side-effect — so making it explicit here also fixes that latent gap.
# Membership-guarded so a re-import (or a vendored-file revert) never raises
# "already registered".
def _reregister(registry, name, cls):
    if registry.get(name) is None:
        registry.register_module(name=name, module=cls)


_reregister(DATASETS, "LUCiDEventSSLDataset", MultiModalEventDataset)

# KEEP-IN-PIMM: the DDP dataloader (imports pimm.utils.comm / env).
from .dataloader import MultiDatasetDataloader

__all__ = [
    "DATASETS", "TRANSFORMS", "Compose", "build_dataset",
    "collate_fn", "point_collate_fn", "inseg_collate_fn",
    "DefaultDataset", "ConcatDataset",
    "PILArNetH5Dataset", "JAXTPCDataset", "LUCiDDataset",
    "MultiModalEventDataset",
    "MultiDatasetDataloader",
    "compute_anchors", "ANCHOR_DEFAULT_CFG",
]
