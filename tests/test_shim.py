"""Smoke test: the de-fork shim re-exports the data layer from pimm-data.

Guards the deletion of the vendored data layer — if the libs/pimm-data submodule
(or its editable install) resolves the wrong tree, or a re-export is missing,
this fails loudly instead of at training time. (Replaces the old
test_jaxtpc_dataset / test_lucid_dataset which imported the now-deleted
pimm.datasets.{jaxtpc_dataset,lucid_dataset,transform,utils}.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_shim_reexports_resolve():
    import pimm.datasets as d
    for name in ("DATASETS", "TRANSFORMS", "build_dataset", "collate_fn",
                 "point_collate_fn", "JAXTPCDataset", "LUCiDDataset",
                 "PILArNetH5Dataset", "MultiModalEventDataset",
                 "MultiDatasetDataloader"):
        assert getattr(d, name, None) is not None, f"shim missing {name}"


def test_defork_transforms_registered():
    # the de-fork additions (and the core transforms) must resolve via the registry
    import pimm.datasets as d
    for t in ("ApplyToStream", "RemapSegment", "Collect", "GridSample", "ToTensor"):
        assert d.TRANSFORMS.get(t) is not None, f"transform {t} not registered"
