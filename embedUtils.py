#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper utility for computing and caching SAM image embeddings and mask features on-the-fly.

This module provides an EmbeddingManager class that handles the computation
and caching of features needed for the PairRouter, abstracting away the file I/O
for use within a larger processing script.
"""

import os
import numpy as np
import cv2
import torch
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, Tuple

# -------------------------- Feature Computation Functions -------------------------- #
# These functions are adapted from your batch script to work on in-memory objects.

@torch.no_grad()
def compute_sam_embeddings(predictor: SamPredictor, image_rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes full and pooled SAM embeddings for a single image."""
    predictor.set_image(image_rgb)
    full_embed = predictor.get_image_embedding()  # Shape: (1, 256, 64, 64) for ViT-H
    pooled_embed = full_embed.mean(dim=(2, 3)).squeeze(0)  # Shape: (256,)
    return full_embed, pooled_embed

def compute_cheap_mask_features(mask_bin: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """Computes a vector of simple, non-SAM-based mask features."""
    m = (mask_bin > 127).astype(np.uint8)
    H, W = m.shape
    area = m.mean()

    ys, xs = np.where(m)
    if xs.size == 0:
        bbox_fill, aspect = 0.0, 0.0
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        bbox_fill = m.sum() / float(bw * bh)
        aspect = bw / float(bh)
    
    ph = np.abs(m[:, 1:] - m[:, :-1]).sum() if W > 1 else 0
    pv = np.abs(m[1:, :] - m[:-1, :]).sum() if H > 1 else 0
    perim = float(ph + pv)
    perim_norm = perim / max(H * W, 1)
    compactness = (4.0 * np.pi * m.sum()) / (perim**2 + 1e-6) if perim > 0 else 0.0

    num_labels, _ = cv2.connectedComponents(m, connectivity=4)
    num_components = float(max(0, num_labels - 1))

    return np.array([area, bbox_fill, aspect, perim_norm, compactness, num_components], dtype=np.float32)

def compute_sam_mask_features(mask_bin: np.ndarray, full_embed: torch.Tensor) -> np.ndarray:
    """Computes SAM-aware features for a mask using the full image embedding."""
    m = (mask_bin > 127).astype(np.uint8)
    
    # Resize mask to match embedding dimensions proportionately
    Hf, Wf = full_embed.shape[-2:]
    Hm, Wm = m.shape
    if Hm >= Wm:
        rW = int(Hf * Wm / max(1, Hm)); rH = Hf
    else:
        rH = int(Wf * Hm / max(1, Wm)); rW = Wf
    
    m_resized = cv2.resize(m, (rW, rH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    m_res_t = torch.from_numpy(m_resized).to(full_embed.device)

    with torch.no_grad():
        E_crop = full_embed[0, :, :rH, :rW]  # Crop embedding to match resized mask
        token = (E_crop * m_res_t).sum(dim=(1, 2)) / (m_res_t.sum() + 1e-6)
        token_normalized = torch.nn.functional.normalize(token, dim=0)
    
    return token_normalized.cpu().numpy().astype(np.float32)


# -------------------------- Main Manager Class -------------------------- #

class EmbeddingManager:
    """
    Manages the on-the-fly computation and caching of embeddings and features.
    """
    def __init__(self, sam_checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda:0"):
        print("Initializing EmbeddingManager and loading SAM model...")
        self.device = device
        self.sam_predictor = self._build_predictor(sam_checkpoint_path, model_type, device)
        
        # In-memory caches to avoid re-computation within a single run
        self.image_embed_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.mask_feature_cache: Dict[Tuple[str, int], np.ndarray] = {}
        print("EmbeddingManager ready.")

    @classmethod
    def from_predictor(cls, predictor: SamPredictor, device: str = "cuda:0") -> "EmbeddingManager":
        """
        Alternate constructor to reuse an existing SamPredictor (avoids loading SAM twice).
        """
        obj = cls.__new__(cls)
        obj.device = device
        obj.sam_predictor = predictor
        obj.image_embed_cache = {}
        obj.mask_feature_cache = {}
        print("EmbeddingManager ready (reusing provided SamPredictor).")
        return obj

    def _build_predictor(self, ckpt: str, model_type: str, device: str) -> SamPredictor:
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device=device)
        return SamPredictor(sam)

    def seed_image_embeddings(
        self,
        image_path: str,
        full_embed: torch.Tensor,
        pooled_embed: torch.Tensor,
    ) -> None:
        """
        Seed the cache so get_image_embeddings() is a no-op for this image.

        Expected shapes:
          - full_embed:  (1, C, H', W')
          - pooled_embed: (C,) or (1, C)
        """
        if pooled_embed.dim() == 2 and pooled_embed.size(0) == 1:
            pooled_embed = pooled_embed.squeeze(0)
        self.image_embed_cache[image_path] = (full_embed, pooled_embed)

    def get_image_embeddings(self, image_path: str, image_rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves (or computes and caches) the full and pooled SAM embeddings for an image.
        
        Args:
            image_path (str): Unique identifier for the image (used as cache key).
            image_rgb (np.ndarray): The loaded RGB image as a numpy array.
            
        Returns:
            A tuple containing (full_embedding, pooled_embedding).
        """
        if image_path in self.image_embed_cache:
            return self.image_embed_cache[image_path]
        
        # Compute and cache
        full, pooled = compute_sam_embeddings(self.sam_predictor, image_rgb)
        self.image_embed_cache[image_path] = (full, pooled)
        return full, pooled

    def get_mask_features(self, image_path: str, image_rgb: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        """
        Retrieves (or computes and caches) the feature vector for a coarse mask.
        This combines both "cheap" and "SAM-aware" features.
        
        Args:
            image_path (str): Path to the original image.
            image_rgb (np.ndarray): The loaded RGB image.
            coarse_mask (np.ndarray): The coarse mask (H, W) with values in {0, 255}.

        Returns:
            A combined feature vector as a numpy array.
        """
        # Create a unique key for the mask based on its content hash
        mask_hash = hash(coarse_mask.tobytes())
        cache_key = (image_path, mask_hash)

        if cache_key in self.mask_feature_cache:
            return self.mask_feature_cache[cache_key]
        
        # 1. Compute cheap features
        cheap_feats = compute_cheap_mask_features(coarse_mask, image_rgb)
        
        # 2. Compute SAM-aware features
        full_embed, _ = self.get_image_embeddings(image_path, image_rgb)
        sam_feats = compute_sam_mask_features(coarse_mask, full_embed)
        
        # 3. Combine and cache
        combined_features = np.concatenate([sam_feats, cheap_feats]).astype(np.float32)
        self.mask_feature_cache[cache_key] = combined_features
        return combined_features
