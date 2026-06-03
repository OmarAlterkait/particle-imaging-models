"""pimm-side SSL/pretraining transforms: registration + smoke.

These pretraining-recipe transforms (multi-view generation, MAE masking, anchor
mining, instance-target parsing) were moved out of pimm-data into
pimm/datasets/transforms.py; they register into the SHARED pimm_data.TRANSFORMS
via the eager import in pimm/datasets/__init__.py. This guards that the move +
eager registration resolve (the failure mode is a config referencing e.g.
type="MultiViewGenerator" raising KeyError at dataset-build time).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SSL_TRANSFORMS = (
    "ContrastiveViewsGenerator", "MultiViewGenerator",
    "MixedScaleGeometryMultiViewGenerator", "ComputeAnchors",
    "InstanceParser", "HierarchicalMaskGenerator",
)


def test_ssl_transforms_registered():
    import pimm.datasets as d
    for t in _SSL_TRANSFORMS:
        assert d.TRANSFORMS.get(t) is not None, \
            f"{t} not registered into the shared pimm_data.TRANSFORMS"


def test_mixed_scale_multiview_smoke():
    # Moved from pimm-data tests/test_transform_v3_vertex.py.
    import pimm.datasets as d
    np.random.seed(3)
    rng = np.random.default_rng(0)
    n = 400
    coord = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    energy = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    cls = d.TRANSFORMS.get("MixedScaleGeometryMultiViewGenerator")
    gen = cls(
        fine_local_view_num=2,
        fine_local_view_scale=(0.05, 0.1),
        fine_center_mode="geometry",
        global_view_num=2,
        global_view_scale=(0.5, 1.0),
        local_view_num=4,
        local_view_scale=(0.2, 0.4),
        view_keys=("coord", "energy"),
        max_size=n,
    )
    out = gen({"coord": coord.copy(), "energy": energy.copy()})
    assert "global_coord" in out and "local_coord" in out
    assert "global_offset" in out and "local_offset" in out
    assert out["global_offset"][-1] == out["global_coord"].shape[0]
    assert out["local_offset"][-1] == out["local_coord"].shape[0]
