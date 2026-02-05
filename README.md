# Moving Target Detection

A moving target detection system implemented using OpenCV. The pipeline detects motion in video streams and draws bounding boxes around moving objects by combining background subtraction, optical flow filtering, and motion history analysis.

## Overview
This project processes video frames to detect moving objects in real time. Multiple motion estimation and filtering techniques are combined to reduce noise and produce stable detections.

## Pipeline
1. Background subtraction using MOG2
2. Optical flow filtering to remove low-motion noise
3. Morphological operations for mask cleanup
4. Motion History Image (MHI) for temporal consistency
5. Contour extraction and filtering
6. Merging nearby detections
7. Rendering bounding boxes on output frames

## Features
- Robust moving object detection
- Optical flow based motion filtering
- Temporal motion aggregation using MHI
- Noise reduction via morphological processing
- Bounding box merging for cleaner detections
- Output video generation with annotations

## Requirements
Install dependencies:

bash
pip install opencv-python numpy

## Run
Place your input video inside the `videos` folder and run:

```bash
python src/mtd.py
