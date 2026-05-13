# CA-CFF-UNet

This repository contains the core code for automatic alluvial fan extraction based on remote sensing images.

## Project Description

This project focuses on automatic extraction of alluvial fans from multi-source remote sensing data. The input data include RGB optical images, SRTM DEM, slope and hillshade layers. Based on the U-Net framework, coordinate attention and cross-scale feature fusion modules are introduced to improve the representation of directional spatial information and multi-scale geomorphic features.

## Main Files

- `dataset_optimized.py`: Dataset loading and multi-channel patch organization
- `model_ca_cff_unet.py`: CA-CFF-UNet model structure
- `losses.py`: Segmentation loss functions
- `train_ca_cff_unet.py`: Model training script
- `train_ca_cff_finetune.py`: Target-domain fine-tuning script
- `predict_ca_cff_unet.py`: Whole-image prediction script
- `stitch_patches.py`: Patch stitching and post-processing script

## Workflow

1. Prepare multi-source raster data, including RGB image, DEM, slope and hillshade.
2. Generate multi-channel patches and corresponding binary masks.
3. Train CA-CFF-UNet using patch samples.
4. Fine-tune the model on target-domain samples.
5. Predict full remote sensing tiles using sliding windows.
6. Apply voting-based stitching and morphological post-processing.

## Note

Large remote sensing images, DEM data, patch datasets, model weights and intermediate results are not included in this repository.
