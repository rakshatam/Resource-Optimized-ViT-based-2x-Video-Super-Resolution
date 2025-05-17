# Resource-Optimized-ViT-based-2x-Video-Super-Resolution
This project implements a video super-resolution pipeline using a **Vision Transformer (ViT)** architecture with **PixelShuffle** upsampling. The model is trained to upscale low-resolution (LR) video to high-resolution (HR) video, demonstrating the capabilities of Transformer networks in video enhancement tasks. This project was developed and trained on consumer-grade hardware with limited VRAM, showcasing efficient implementation strategies.

---

## Overview

This repository contains the code for training and deploying a **2x video super-resolution** model based on a Vision Transformer. The pipeline consists of the following key stages:

1.  **Low-Resolution Input:** Takes a low-resolution video as input.
2.  **Frame Extraction:** Individual frames are extracted from the input video.
3.  **Super-Resolution with ViT:** Each frame is processed by the trained Vision Transformer model, which learns to generate a higher-resolution version by leveraging global context and detailed feature extraction. The model utilizes a patch embedding layer, a Transformer encoder, and a PixelShuffle layer for efficient upsampling.
4.  **High-Resolution Output:** The super-resolved frames are then reassembled to produce the final high-resolution video.

This project highlights the effectiveness of ViT architectures for video enhancement and demonstrates techniques for training such models on resource-constrained hardware like an **NVIDIA RTX 3050 laptop with 4GB VRAM**.

---

## Pipeline Workflow

The pipeline involves two main parts: **Training** and **Deployment (Inference)**.

### 1. Training Pipeline

The training script (`train_model.py` - *consider renaming your training script for clarity*) follows these steps:

1.  **Dataset Preparation:** Loads the DIV2K dataset (or your chosen dataset) with paired low-resolution (bicubic downscaled) and high-resolution images.
2.  **Patch Extraction:** Extracts smaller patches from the LR and HR images to facilitate training on limited VRAM. Multiple patches are sampled per image to increase the training data size.
3.  **Model Definition:** Defines the `ViTSuperResolutionModel`, which includes:
    * A convolutional layer for initial patch embedding.
    * A `CheckpointedTransformer` (utilizing `torch.utils.checkpoint`) for memory-efficient Transformer encoding.
    * Convolutional layers and a `PixelShuffle` layer for 2x upsampling.
4.  **Training Loop:** Trains the model using L1 loss and the AdamW optimizer with a learning rate scheduler. Key techniques employed for efficiency include:
    * **Gradient Accumulation:** Simulates larger batch sizes on limited VRAM.
    * **Automatic Mixed Precision (AMP):** Reduces memory footprint and accelerates computation by using lower-precision floating-point numbers where appropriate.
    * **Memory Checkpointing:** Reduces GPU memory usage during backpropagation by recomputing parts of the graph.
5.  **Validation:** Periodically evaluates the model on a validation set to monitor performance (using PSNR as a key metric) and prevent overfitting.
6.  **Model Saving:** Saves the best-performing model weights (`best_model.pth`) based on the validation PSNR. Checkpoints are also saved periodically.

### 2. Deployment (Inference) Pipeline

The deployment script (`deploy_model.py` - *consider renaming your deployment script for clarity*) takes an input video and outputs an upscaled video:

1.  **Model Loading:** Loads the trained weights (`best_model.pth`) into an instance of the `ViTVideoSR` model architecture.
2.  **Video Input:** Reads the input low-resolution video using OpenCV (`cv2`).
3.  **Frame-by-Frame Processing:** Iterates through each frame of the input video:
    * Converts the frame to a PyTorch tensor.
    * Performs inference using the loaded ViT-SR model (within a `torch.no_grad()` context and with AMP enabled for speed and efficiency).
    * Clamps the output pixel values to the valid range [0, 1].
    * Converts the super-resolved tensor back to an OpenCV image format.
4.  **Video Output:** Writes the sequence of super-resolved frames to a new high-resolution video file using OpenCV.
5.  **Progress Tracking:** Provides a real-time indication of the processing progress (frames processed).

---

## Input and Output Examples

### Input

The pipeline accepts standard video files (e.g., `.mp4`, `.avi`) as input. Below is an example of a low-resolution input frame (480x270):

![Low-Resolution Input Frame](pic 1.png)

### Output

The pipeline generates a 2x upscaled video file with double the width and height of the input. Here's an example of a corresponding super-resolved output frame (960x540):

![High-Resolution Output Frame](pic 2.png)

**Note:** The visual improvement will depend on the complexity of the original video and the quality achieved during training (e.g., your reported PSNR of around 30 dB).

## Model Training

To train the model yourself:

1.  Ensure you have the DIV2K dataset (or your chosen dataset) organized correctly.
2.  Modify the `TRAIN_HR_DIR`, `TRAIN_LR_DIR`, `VALID_HR_DIR`, and `VALID_LR_DIR` variables in the training script (`train_model.py` or similar) to point to your dataset locations.
3.  Run the training script:
    ```bash
    python train_model.py
    ```
    *Training can take a significant amount of time depending on your hardware and the number of epochs.*

---

## Hardware Used

The model was trained and tested on a laptop with the following specifications:

* **GPU:** NVIDIA GeForce RTX 3050 (4GB VRAM)
* **CPU:** [Your CPU Model, e.g., Intel Core i7-11800H]
* **RAM:** [Your RAM Amount, e.g., 16GB DDR4]

The project demonstrates the feasibility of training relatively complex deep learning models for video enhancement even on consumer-level hardware with careful resource management.

---

## Results and Observations

* The trained model achieved a peak **PSNR of approximately 30 dB** on the validation set after 50 epochs of training. Further training may lead to higher PSNR values.
* The deployment pipeline successfully upscales video while managing memory constraints.
* Visual inspection of the upscaled video shows improved detail and sharpness compared to the low-resolution input, although the extent of the improvement depends on the original content and the model's training.
* The output video exhibits an increased bitrate, which is expected for higher resolution content (e.g., from ~900 kbps for 480p to ~2 Mbps for 960p).

---

## Future Work

* Train the model for a significantly larger number of epochs to potentially achieve higher PSNR and visual quality.
* Experiment with different ViT model configurations (e.g., varying `MODEL_EMBED_DIM`, `MODEL_NUM_HEADS`, `MODEL_NUM_LAYERS`).
* Explore the integration of perceptual loss functions to improve the perceptual quality of the upscaled video.
* Investigate the use of other datasets for training and generalization.
* Optimize the deployment pipeline for real-time performance on various hardware platforms.
