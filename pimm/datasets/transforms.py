"""pimm-owned SSL / pretraining-recipe transforms (moved from pimm-data).

These are pretraining-recipe-specific — multi-view generation, MAE-style patch
masking, anchor mining, and instance-target parsing — so they live with the
models in pimm rather than in the standalone data layer. They register into the
shared ``pimm_data.TRANSFORMS`` registry on import (this module is imported
eagerly by ``pimm/datasets/__init__.py``). See pimm-data
``docs/ADR-pimm-data-package-boundary.md`` and ``IMPLEMENTATION-boundary-refactor.md``.
"""
from typing import Optional
import copy

import numpy as np
from scipy.spatial import cKDTree

# Shared registry + Compose stay in pimm-data; we register INTO them.
from pimm_data.transform import TRANSFORMS, Compose
from .anchors import compute_anchors, ANCHOR_DEFAULT_CFG


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class MultiViewGenerator(object):
    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        center_height_scale=(0, 1),
        shared_global_view=False,
        center_sampling="random",  # or cnms
        center_sampling_kwargs=None,
        view_keys=("coord", "origin_coord", "color", "normal"),
        # Anchor-biased sampling
        anchor_bias_ratio=0.6,
        anchor_radius_scale=1.5,
        anchor_keys=("endpoints", "branches_track", "branches_shower", "bragg"),
    ):
        self.global_view_num = global_view_num
        self.global_view_scale = global_view_scale
        self.local_view_num = local_view_num
        self.local_view_scale = local_view_scale
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        assert "coord" in view_keys
        self.center_sampling = center_sampling
        self.center_sampling_kwargs = center_sampling_kwargs
        # Anchors
        self.anchor_bias_ratio = anchor_bias_ratio
        self.anchor_radius_scale = anchor_radius_scale
        self.anchor_keys = anchor_keys

    def get_view(self, point, center, scale, size_override: Optional[int] = None):
        coord = point["coord"]
        max_size = min(self.max_size, coord.shape[0])
        if max_size <= 0:
            raise ValueError("Cannot generate a view from an empty point cloud")
        size = int(np.random.uniform(*scale) * max_size) if size_override is None else int(size_override)
        size = max(1, min(max_size, size))
        index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
        view = dict(index=index)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][index]

        if "index_valid_keys" in point.keys():
            # inherit index_valid_keys from point
            view["index_valid_keys"] = point["index_valid_keys"]
        return view

    def get_center(self, coord, mask=None):
        if mask is None:
            possible_centers = coord
        else:
            possible_centers = coord[np.where(mask)[0]]
        if self.center_sampling == "cnms":
            from cnms import cnms
            possible_centers, _, _ = cnms(possible_centers, **self.center_sampling_kwargs)
        return possible_centers[np.random.choice(possible_centers.shape[0])]

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
        # get major global view
        major_center = coord[np.random.choice(np.where(center_mask)[0])]
        major_view = self.get_view(point, major_center, self.global_view_scale)
        major_coord = major_view["coord"]
        # get global views: restrict the center of left global view within the major global view
        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]

        global_views = [major_view] + global_views

        # get local views: restrict the center of local view within the major global view
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        # Prepare anchor pool if available (exclude LEDs)
        anchors_pool = []
        if isinstance(data_dict.get("anchors"), dict):
            for k in self.anchor_keys:
                if k == "led":
                    continue
                v = data_dict["anchors"].get(k)
                if v is not None and len(v) > 0:
                    anchors_pool.append(v)
        anchors_pool = np.concatenate(anchors_pool, axis=0) if len(anchors_pool) > 0 else np.zeros((0,3), dtype=np.float32)

        # Map anchors to nearest point inside major view to keep locality consistent
        kd_major = cKDTree(major_coord) if major_coord.shape[0] > 0 else None
        # Estimate size override for anchor crops: approximate radius scaling via cubic relation
        # size' ~= size * (radius_scale^3)
        size_base = int(np.mean([np.random.uniform(*self.local_view_scale) * min(self.max_size, coord.shape[0]) for _ in range(4)]))
        size_override = int(max(8, min(self.max_size, size_base * (self.anchor_radius_scale ** 3))))

        # Determine counts
        num_anchor_locals = int(np.ceil(self.local_view_num * float(self.anchor_bias_ratio))) if anchors_pool.shape[0] > 0 else 0
        num_random_locals = self.local_view_num - num_anchor_locals

        # Anchor-centered locals
        for i in range(num_anchor_locals):
            if sum(~cover_mask) == 0:
                cover_mask[:] = False
            if anchors_pool.shape[0] == 0:
                break
            aidx = np.random.randint(0, anchors_pool.shape[0])
            acoord = anchors_pool[aidx]
            # Project to nearest major point to keep within major global view
            if kd_major is not None and kd_major.n > 0:
                _, nn = kd_major.query(acoord, k=1)
                center = major_coord[nn]
            else:
                center = acoord
            local_view = self.get_view(
                point=data_dict,
                center=center,
                scale=self.local_view_scale,
                size_override=size_override,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # Uniform random locals
        for i in range(num_random_locals):
            if sum(~cover_mask) == 0:
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # augmentation and concat
        view_dict = {}
        for global_view in global_views:
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]
        view_dict["global_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["global_coord"]]
        )
        for local_view in local_views:
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]
        view_dict["local_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["local_coord"]]
        )
        for key in view_dict.keys():
            if "offset" not in key:
                view_dict[key] = np.concatenate(view_dict[key], axis=0)
        data_dict.update(view_dict)
        return data_dict


@TRANSFORMS.register_module()
class MixedScaleGeometryMultiViewGenerator(MultiViewGenerator):
    """Multi-view generator with normal coarse locals plus fine local crops.

    Fine local crop centers can be sampled uniformly or from a simple local PCA
    directional-complexity score. This keeps the SSL objective unchanged while
    changing which local regions feed the local-global loss.
    """

    def __init__(
        self,
        fine_local_view_num=3,
        fine_local_view_scale=(0.01, 0.04),
        fine_center_mode="geometry",
        fine_center_top_frac=0.05,
        fine_center_k=24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0 <= fine_local_view_num <= self.local_view_num
        assert fine_center_mode in ("geometry", "random")
        self.fine_local_view_num = int(fine_local_view_num)
        self.fine_local_view_scale = fine_local_view_scale
        self.fine_center_mode = fine_center_mode
        self.fine_center_top_frac = float(fine_center_top_frac)
        self.fine_center_k = int(fine_center_k)

    @staticmethod
    def _directional_complexity(coord, k):
        coord = np.asarray(coord, dtype=np.float32)
        n = coord.shape[0]
        if n < 4:
            return np.zeros(n, dtype=np.float32)
        k_eff = min(int(k) + 1, n)
        tree = cKDTree(coord)
        try:
            _, idx = tree.query(coord, k=k_eff, workers=-1)
        except TypeError:
            _, idx = tree.query(coord, k=k_eff)
        if idx.ndim == 1:
            idx = idx[:, None]
        idx = idx[:, 1:]
        if idx.shape[1] < 3:
            return np.zeros(n, dtype=np.float32)
        neigh = coord[idx]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / centered.shape[1]
        eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        return (eig[:, 1] / (eig[:, 2] + 1.0e-8)).astype(np.float32)

    def _geometry_pool(self, coord, major_index):
        if self.fine_center_mode == "random":
            return major_index
        score = self._directional_complexity(coord, self.fine_center_k)
        n_top = max(1, int(np.ceil(score.shape[0] * self.fine_center_top_frac)))
        top_index = np.argpartition(score, -n_top)[-n_top:]
        in_major = np.zeros(coord.shape[0], dtype=bool)
        in_major[major_index] = True
        pool = top_index[in_major[top_index]]
        return pool if pool.shape[0] > 0 else major_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)

        major_center = coord[np.random.choice(np.where(center_mask)[0])]
        major_view = self.get_view(point, major_center, self.global_view_scale)
        major_coord = major_view["coord"]

        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]
        global_views = [major_view] + global_views

        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        fine_pool = self._geometry_pool(coord, major_view["index"])

        for _ in range(self.fine_local_view_num):
            center = coord[fine_pool[np.random.randint(fine_pool.shape[0])]]
            local_views.append(
                self.get_view(
                    point=data_dict,
                    center=center,
                    scale=self.fine_local_view_scale,
                )
            )

        num_random_locals = self.local_view_num - self.fine_local_view_num
        for _ in range(num_random_locals):
            if sum(~cover_mask) == 0:
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        view_dict = {}
        for global_view in global_views:
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]
        view_dict["global_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["global_coord"]]
        )
        for local_view in local_views:
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]
        view_dict["local_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["local_coord"]]
        )
        for key in view_dict.keys():
            if "offset" not in key:
                view_dict[key] = np.concatenate(view_dict[key], axis=0)
        data_dict.update(view_dict)
        return data_dict


@TRANSFORMS.register_module()
class ComputeAnchors(object):
    """Compute anchors once per event and attach to data_dict['anchors'].

    Args:
        cfg: dict overriding anchor defaults
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or dict()

    def __call__(self, data_dict):
        if compute_anchors is None:
            return data_dict
        if "coord" not in data_dict or "energy" not in data_dict:
            return data_dict
        xyz = data_dict["coord"].astype(np.float32)
        # energy may be (N,) or (N,1); use (N,)
        e = data_dict["energy"]
        if e.ndim > 1 and e.shape[-1] == 1:
            e = e.reshape(-1)
        # Merge defaults with overrides
        cfg = dict(ANCHOR_DEFAULT_CFG)
        cfg.update(self.cfg)
        anchors = compute_anchors(xyz=xyz, energy=e, is_shower_like=None, cfg=cfg)
        data_dict["anchors"] = anchors
        # Exclude LEDs from being used inadvertently elsewhere
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(
        self,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        compute_axis_stats=False,
        axis_min_points=5,
        axis_eps=1e-6,
        axis_default=(1.0, 0.0, 0.0),
        axis_normalize_half_extent=True,
    ):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.compute_axis_stats = bool(compute_axis_stats)
        self.axis_min_points = max(int(axis_min_points), 1)
        self.axis_eps = float(axis_eps)
        axis_default = np.asarray(axis_default, dtype=np.float32)
        if axis_default.shape != (3,):
            raise ValueError("axis_default must have shape (3,)")
        axis_norm = np.linalg.norm(axis_default)
        if axis_norm <= 0:
            raise ValueError("axis_default must be non-zero")
        self.axis_default = axis_default / axis_norm
        self.axis_normalize_half_extent = bool(axis_normalize_half_extent)

    def __call__(self, data_dict):
        coord = np.asarray(data_dict["coord"])
        coord_dtype = coord.dtype
        # ensure 1D arrays for correct boolean indexing
        segment = data_dict["segment"]
        if isinstance(segment, np.ndarray):
            segment = segment.reshape(-1)
        else:
            segment = np.asarray(segment).reshape(-1)
        instance = data_dict["instance"]
        if isinstance(instance, np.ndarray):
            instance = instance.reshape(-1)
        else:
            instance = np.asarray(instance).reshape(-1)
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3), dtype=coord_dtype) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8), dtype=coord_dtype) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        if self.compute_axis_stats:
            axis_default = self.axis_default.astype(coord_dtype, copy=False)
            axis = np.tile(axis_default, (coord.shape[0], 1))
            axis_coord = np.zeros(coord.shape[0], dtype=coord_dtype)
            axis_coord_normalized = np.zeros(coord.shape[0], dtype=coord_dtype)
            axis_length = np.zeros(coord.shape[0], dtype=coord_dtype)
            axis_weight = np.zeros(coord.shape[0], dtype=coord_dtype)
        else:
            axis = axis_coord = axis_coord_normalized = axis_length = axis_weight = None

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid.astype(coord_dtype, copy=False)
            bbox_row = np.concatenate([bbox_center, bbox_size, bbox_theta, bbox_class])
            bbox[instance_id] = bbox_row.astype(coord_dtype, copy=False)

            if self.compute_axis_stats:
                point_count = coord_.shape[0]
                valid_axis = False
                axis_vec = axis_default
                axis_coord_local = np.zeros(point_count, dtype=coord_dtype)
                axis_coord_norm_local = np.zeros(point_count, dtype=coord_dtype)
                axis_length_value = 0.0
                if point_count >= self.axis_min_points:
                    centered = coord_.astype(np.float32, copy=False) - bbox_centroid.astype(np.float32, copy=False)
                    if np.linalg.norm(centered, axis=1).max() > self.axis_eps:
                        cov = centered.T @ centered
                        cov /= max(point_count, 1)
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        principal_index = int(np.argmax(eigvals))
                        principal_val = float(eigvals[principal_index])
                        principal_vec = eigvecs[:, principal_index].astype(np.float32, copy=False)
                        principal_norm = float(np.linalg.norm(principal_vec))
                        if principal_norm > self.axis_eps and principal_val > self.axis_eps:
                            axis_vec = principal_vec / principal_norm
                            projections = centered @ axis_vec
                            max_proj = float(projections.max())
                            min_proj = float(projections.min())
                            axis_length_value = max_proj - min_proj
                            if axis_length_value > self.axis_eps:
                                valid_axis = True
                                axis_coord_local = projections.astype(coord_dtype, copy=False)
                                denom = axis_length_value * 0.5 if self.axis_normalize_half_extent else axis_length_value
                                denom = float(denom) + self.axis_eps
                                axis_coord_norm_local = (axis_coord_local / denom).astype(coord_dtype, copy=False)
                if not valid_axis:
                    axis_vec = axis_default
                    axis_coord_local.fill(0.0)
                    axis_coord_norm_local.fill(0.0)
                    axis_length_value = 0.0
                axis[mask_] = axis_vec.astype(coord_dtype, copy=False)
                axis_coord[mask_] = axis_coord_local
                axis_coord_normalized[mask_] = axis_coord_norm_local
                axis_length[mask_] = axis_length_value
                axis_weight_value = 1.0 if valid_axis else 0.0
                axis_weight[mask_] = axis_weight_value

        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        if self.compute_axis_stats:
            data_dict["instance_axis"] = axis
            data_dict["instance_axis_coord"] = axis_coord
            data_dict["instance_axis_coord_normalized"] = axis_coord_normalized
            data_dict["instance_axis_length"] = axis_length
            data_dict["instance_axis_weight"] = axis_weight
            data_dict["instance_axis_coord_weight"] = axis_weight
        return data_dict


@TRANSFORMS.register_module()
class HierarchicalMaskGenerator(object):
    """
    Generate hierarchical masks for MAE style pretraining.

    Points are grouped into patches at patch_size granularity, then a fraction
    are randomly masked. The visible points go to the encoder, while masked
    patch information is stored for the decoder to reconstruct.
    
    Uses same grid-based hashing as GridSample to ensure exact alignment with
    PTv3's coarsest features after hierarchical pooling.
    
    Important: Centroids are grid cell centers (not point means) to ensure
    proper 1:1 correspondence between patches and coarse encoder features.
    """

    def __init__(
        self,
        patch_size: float = 0.016,
        mask_ratio: float = 0.6,
        points_per_patch: int = 128,
        min_points_per_patch: int = 0,
        view_keys: tuple = ("coord", "origin_coord", "energy"),
    ):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.points_per_patch = points_per_patch
        self.min_points_per_patch = min_points_per_patch
        self.view_keys = view_keys

    @staticmethod
    def fnv_hash_vec(arr):
        """FNV64-1A hash for grid coordinates"""
        assert arr.ndim == 2
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        n_points = coord.shape[0]

        if n_points == 0:
            data_dict["hmae_valid"] = False
            return data_dict

        # grid coordinates aligned to patch_size, matching PTv3's grid structure
        # matches: floor((coord - coord.min()) / patch_size) after 4x stride-2 poolings
        coord_min = coord.min(axis=0)
        grid_coord = np.floor((coord - coord_min) / self.patch_size).astype(np.int64)

        # spatial hash using FNV (same as GridSample for consistency)
        patch_ids = self.fnv_hash_vec(grid_coord)

        # unique patches and assignment of each point to a patch
        unique_patches, inverse_indices, patch_counts = np.unique(
            patch_ids, return_inverse=True, return_counts=True
        )
        n_patches = len(unique_patches)

        # keep only patches with enough points
        valid_patch_mask = patch_counts >= self.min_points_per_patch
        valid_patch_indices = np.where(valid_patch_mask)[0]
        n_valid_patches = len(valid_patch_indices)

        if n_valid_patches < 2:
            data_dict["hmae_valid"] = False
            return data_dict

        # choose which patches are masked vs visible
        n_mask = max(1, int(n_valid_patches * self.mask_ratio))
        n_visible = n_valid_patches - n_mask

        perm = np.random.permutation(n_valid_patches)
        masked_patch_local_idx = perm[:n_mask]
        visible_patch_local_idx = perm[n_mask:]

        masked_patch_idx = valid_patch_indices[masked_patch_local_idx]
        visible_patch_idx = valid_patch_indices[visible_patch_local_idx]

        # vectorized visible mask: map patch index -> visible flag
        is_visible_patch = np.zeros(n_patches, dtype=bool)
        is_visible_patch[visible_patch_idx] = True
        visible_mask = is_visible_patch[inverse_indices]

        # extract visible data for encoder
        visible_data = {}
        for key in self.view_keys:
            if key in data_dict:
                visible_data[key] = data_dict[key][visible_mask]

        # sort points once by patch index for efficient masked patch processing
        # this avoids repeatedly doing (inverse_indices == patch_idx) per patch
        order = np.argsort(inverse_indices)
        sorted_coord = coord[order]
        sorted_grid_coord = grid_coord[order]
        has_energy = "energy" in data_dict
        if has_energy:
            sorted_energy = data_dict["energy"][order]

        # build CSR style offsets from patch_counts
        # patch_offsets[j] .. patch_offsets[j+1] is the slice for patch j
        patch_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(patch_counts, dtype=np.int64)]
        )

        # masked patch targets
        masked_centroids = []
        masked_target_coords = []  # list of (Ni, 3)
        masked_target_energy = []  # list of (Ni, 1) if available
        masked_point_counts = []

        norm_factor = self.patch_size / 2.0

        for patch_idx in masked_patch_idx:
            start = patch_offsets[patch_idx]
            end = patch_offsets[patch_idx + 1]
            if end <= start:
                continue  # should not happen if min_points_per_patch checked

            patch_coord = sorted_coord[start:end]
            patch_grid_coord = sorted_grid_coord[start:end]
            
            # centroid = geometric center of grid cell (not point mean)
            # all points in this patch share the same grid coordinate
            grid_cell = patch_grid_coord[0]  # same for all points in patch
            centroid = grid_cell * self.patch_size + self.patch_size / 2.0 + coord_min
            masked_centroids.append(centroid)

            # relative coords in [-1, 1]
            rel_coord = (patch_coord - centroid) / norm_factor
            masked_target_coords.append(rel_coord)

            if has_energy:
                patch_energy = sorted_energy[start:end]
                masked_target_energy.append(patch_energy)

            masked_point_counts.append(patch_coord.shape[0])

        if len(masked_centroids) == 0:
            # very rare case: all masked patches dropped for some reason
            data_dict["hmae_valid"] = False
            return data_dict

        # pack results
        data_dict["hmae_valid"] = True

        data_dict["visible_coord"] = visible_data.get("coord", np.array([]))
        data_dict["visible_origin_coord"] = visible_data.get(
            "origin_coord", visible_data.get("coord", np.array([]))
        )

        if "energy" in visible_data:
            data_dict["visible_energy"] = visible_data["energy"]
        else:
            v_n = data_dict["visible_coord"].shape[0]
            data_dict["visible_energy"] = np.zeros((v_n, 1), dtype=np.float32)

        data_dict["masked_centroids"] = np.asarray(masked_centroids, dtype=np.float32)
        data_dict["masked_point_counts"] = np.asarray(masked_point_counts, dtype=np.int64)

        # pack masked patch targets into flattened arrays with offsets (no padding)
        # This replaces the need for HMAECollate
        target_coords_list = []
        target_energy_list = []
        
        for i, coords in enumerate(masked_target_coords):
            target_coords_list.append(coords)
            if i < len(masked_target_energy):
                energy = masked_target_energy[i]
                if energy.ndim == 1:
                    energy = energy[:, None]
                target_energy_list.append(energy)
            else:
                # Create zeros if energy not available for this patch
                target_energy_list.append(np.zeros((coords.shape[0], 1), dtype=np.float32))
        
        # concatenate into flattened arrays
        target_coords_flat = np.concatenate(target_coords_list, axis=0)  # (total_points, 3)
        target_energy_flat = np.concatenate(target_energy_list, axis=0)  # (total_points, 1)
        
        # compute offset per batch sample (not per patch)
        # output just the total point count for this sample
        # batching will convert this to cumulative offsets per batch sample
        total_points = target_coords_flat.shape[0]
        target_offset = np.array([total_points], dtype=np.int64)
        
        data_dict["target_coords"] = target_coords_flat
        data_dict["target_energy"] = target_energy_flat
        data_dict["target_offset"] = target_offset

        data_dict["n_visible_patches"] = n_visible
        data_dict["n_masked_patches"] = len(masked_centroids)

        return data_dict
