# PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting

This repository contains the official code, data, and models for our paper, "PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting" (Submitted to CVPR 2026).

PromptMoE is a model-agnostic framework that reframes segmentation refinement as a dynamic, context-aware prompt-generation task. We introduce a **Dynamic Expert Selector (DES)** to route an image-mask pair to a small, specialized subset of **Image-Informed Prompting (IIP)** experts. These experts (e.g., color, depth, texture cues) generate guidance maps, which are then fused and translated into high-quality, spatially diverse point prompts by our **Prompt-Placement Explorer (PPE)**.

---

## Installation

To set up the environment and download the necessary checkpoints, please follow these steps. This guide assumes you are in a Colab-like environment.

```bash
# 2. Install the Segment Anything Model (SAM) backbone
cd segment-anything
pip install -e .
cd ..

# 3. Install required Python packages
pip install matplotlib transformers scikit-image opencv-python timm pandas
pip install torch torchvision
pip install --upgrade diffusers[torch]

# 4. Download the SAM (ViT-H) checkpoint
mkdir checkpoints/
wget -O checkpoints/sam_vit_h.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Code, Data, and Models

- **Full Source Code:** `PromptMoE.py`  
  Contains the core implementation of the PromptMoE framework, including the IIP expert suite, DES router, and PPE prompter.
- **Demo Notebook:** `demo.ipynb`  
  An interactive notebook demonstrating how to use PromptMoE to refine a coarse segmentation mask on an example image.
- **Trained Model:** `DES_ROUTER.pt`  
  The pre-trained weights for our Dynamic Expert Selector (DES).
- **Paper Results:** `PromptMoE_Paper_Results`  
  All raw results and evaluation code necessary to reproduce the figures and tables in our manuscript.

---

## Main Results

We evaluated PromptMoE against state-of-the-art methods on five diverse benchmark datasets. Our method achieves significant and consistent improvements across semantic, instance, and salient object segmentation tasks.

**Comparative study with state-of-the-art refinement methods across 5 benchmark datasets reporting ΔIoU/ΔBIoU over unrefined base masks. Positive values indicate improvement and negative indicate degradation of mask quality. Our method is highlighted in bold.**

| Method            | BIG               | DAVIS585              | ECSSD                    | MSRA-B                   | VOC                      | Mean ΔIoU / ΔBIoU       |
| :---------------- | :---------------- | :-------------------- | :----------------------- | :----------------------- | :----------------------- | :---------------------- |
| Unrefined         | 78.25 / 70.11     | 80.05 / 83.00         | 81.41 / 70.23            | 75.15 / 61.88            | 66.73 / 60.08            | 76.32 / 69.06           |
| CascadePSP-Fast [8] | +4.29 / +4.18   | −6.21 / −6.99         | −0.99 / −1.21            | −1.01 / −1.01            | +0.42 / −1.30            | −0.70 / −1.27           |
| CascadePSP-Slow [8] | +4.97 / +6.27   | −1.29 / −1.47         | +0.62 / +0.98            | +0.93 / +2.71            | +1.71 / +0.73            | +1.39 / +1.85           |
| SegRefiner-LR [42]  | +2.96 / +2.35   | −15.34 / −18.92       | −13.81 / −21.77          | −10.68 / −17.25          | −3.82 / −6.16            | −8.14 / −12.35          |
| SegRefiner-HR [42]  | **+9.55 / +12.51** | −10.85 / −9.10     | −15.01 / −21.37          | −10.67 / −16.16          | −3.86 / −3.05            | −6.17 / −7.43           |
| DualSight [34]      | +3.89 / +4.63   | +1.48 / −0.57         | +1.99 / +4.50            | +2.08 / +6.79            | +3.29 / +6.29            | +2.55 / +4.33           |
| SAMRefiner [28]     | +6.84 / +9.50   | +3.33 / +2.03         | +5.10 / +9.69            | +4.67 / +10.35           | +7.05 / +9.74            | +5.40 / +8.26           |
| **PromptMoE (Ours)** | **+8.54 / +11.01** | **+3.64 / +2.35** | **+5.99 / +10.67**       | **+5.10 / +10.48**       | **+7.94 / +10.43**       | **+6.24 / +8.99**       |
