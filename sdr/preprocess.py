# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

import os
import cv2
import torch
import numpy as np

annotator_ckpts_path="./"

def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


class HEDdetector:
    def __init__(self,model):
        self.netNetwork = model

    def __call__(self, input_image, safe=False):
        assert input_image.ndim == 3
        H, W, C = input_image.shape
        image_hed = input_image.copy()
        image_hed = np.transpose(image_hed, (2, 0, 1))
        image_hed = image_hed[np.newaxis, :, :, :].astype(np.float32)
        print(image_hed.shape)
        edges = self.netNetwork([image_hed])
        edges = [e.astype(np.float32)[0, 0] for e in edges]
        edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
        edges = np.stack(edges, axis=2)
        edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        if safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return edge
