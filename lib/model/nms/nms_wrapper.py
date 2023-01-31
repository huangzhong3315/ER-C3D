# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from lib.model.utils.config import cfg
# if torch.cuda.is_available():
#     from lib.model.nms.nms_gpu import nms_gpu
from lib.model.nms.nms_cpu import nms_cpu
from lib.model.nms.nms_cpu import softnms_cpu

# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#     if dets.shape[0] == 0:
#         return []
#     # ---numpy version---
#     # original: return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     # ---pytorch version---
#
#     return nms_gpu(dets, thresh) if force_cpu == False else nms_cpu(dets, thresh)

def nms(dets, thresh, force_cpu=True):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []

    # return nms_cpu(dets, thresh)
    return softnms_cpu(dets, thresh)
