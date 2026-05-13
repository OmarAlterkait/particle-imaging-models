from __future__ import annotations

from copy import deepcopy

from pimm.models.builder import MODELS

from .detector_v1m1 import MultiLabelDetector


def _replace_instance_loss(cfg):
    cfg = deepcopy(cfg)
    if isinstance(cfg, dict):
        if cfg.get("type") == "InstanceSegmentationLoss":
            cfg["type"] = "FastInstanceSegmentationLoss"
        return cfg
    if isinstance(cfg, list):
        return [_replace_instance_loss(item) for item in cfg]
    if isinstance(cfg, tuple):
        return tuple(_replace_instance_loss(item) for item in cfg)
    return cfg


@MODELS.register_module("detector-v3m2")
class MultiLabelDetectorV3M2(MultiLabelDetector):
    """Joint Panda detector using the optimized instance-loss implementation."""

    @staticmethod
    def _criteria_cfg(label, criteria, criteria_by_label):
        if criteria_by_label is not None and label in criteria_by_label:
            return _replace_instance_loss(criteria_by_label[label])
        return _replace_instance_loss(criteria)

