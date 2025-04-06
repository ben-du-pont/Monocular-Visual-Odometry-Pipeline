# Monocular Visual Odometry Pipeline

This repository contains an implementation of a monocular visual odometry (VO) pipeline for camera pose estimation and 3D landmark tracking. This pipeline was inspired by the project from a computer vision lecture at ETH and UZH: Vision Algorithms for Mobile Robotics, by Prof. Scaramuzza. It includes several advanced features to improve robustness and accuracy.

![Example Run](media/Example_run.gif)

*Figure: Visualization of the pipeline in action, showing tracked features, camera trajectory, and landmarks.*

## Overview

Visual Odometry (VO) is the process of estimating the egomotion of a camera by analyzing the changes that motion induces on images. This implementation follows a feature-based approach with the following components:

1. **Initialization**: Bootstrap the system by establishing initial 3D landmarks and camera poses
2. **Continuous Operation**:
   - Track keypoints across frames
   - Estimate camera pose using 2D-3D correspondences
   - Triangulate new landmarks to maintain tracking

## Features

### Core Features
- Monocular visual odometry pipeline (no stereo information used)
- KLT feature tracking with forward-backward verification
- P3P RANSAC for robust pose estimation
- Dynamic landmark triangulation with parallax verification
- Visualization of trajectory and landmarks

### Advanced Features
1. **Local Bundle Adjustment** for combating scale drift
   - Optimizes camera poses and 3D landmarks jointly
   - Reduces accumulation of drift over time
   - Implements a sliding window approach for computational efficiency

2. **Keyframe-based Tracking** for improved robustness
   - Identifies keyframes based on feature tracking quality and parallax
   - Uses keyframes as reference for triangulation and scale correction
   - Reduces the risk of drift during quick rotations

3. **Quantitative Feature Tracker Analysis**
   - Compares different feature tracking methods (KLT, SIFT, ORB)
   - Analyzes tracking quality, computational efficiency, and robustness
   - Generates comparative visualizations for evaluation

## Installation

### Requirements
- Python 3.8+
- OpenCV 4.5+
- NumPy
- Matplotlib
- SciPy (for bundle adjustment and KD-tree)

### Setup
```bash
# Clone the repository
git clone https://github.com/ben-du-pont/monocular-visual-odometry-pipeline.git
cd monocular-visual-odometry-pipeline

# Install dependencies
pip install numpy opencv-python matplotlib scipy
```

## Usage

### Running the Pipeline
```bash
python main.py --dataset [kitti|malaga|parking] --path /path/to/dataset
```

### Optional Arguments
- `--start N`: Start processing from frame N (default: 0)
- `--end N`: Stop processing at frame N (default: -1, process all frames)
- `--save`: Save results to output directory
- `--no_display`: Run without visualization
- `--feature_comparison`: Run feature tracker comparison

### Example
```bash
# Run on KITTI dataset with visualization
python main.py --dataset kitti --path /path/to/kitti_dataset --save

# Run on Malaga dataset with feature comparison
python main.py --dataset malaga --path /path/to/malaga_dataset --save --feature_comparison
```

## Pipeline Structure

### Initialization
1. Select two frames with sufficient baseline
2. Detect and track keypoints using KLT through intermediate frames
3. Estimate fundamental matrix using RANSAC to filter outliers
4. Calculate essential matrix from fundamental matrix and calibration
5. Recover relative pose and triangulate initial 3D landmarks

### Continuous Operation
1. Track keypoints from previous to current frame using KLT
2. Filter tracked keypoints using forward-backward verification
3. Estimate current camera pose using P3P RANSAC
4. Update existing landmarks and track candidate keypoints
5. Triangulate new landmarks when sufficient parallax is achieved
6. Apply bundle adjustment periodically to optimize poses and landmarks

## Implementation Details

### State Representation
The state `S_i` at each frame contains:
- `keypoints`: 2D keypoints in the current frame (2xK)
- `landmarks`: Associated 3D landmarks (3xK)
- `candidates`: Candidate keypoints for future triangulation (2xM)
- `first_obs`: First observations of candidate keypoints (2xM)
- `first_poses`: Camera poses at first observations (16xM)

### Keypoint Tracking
- Uses Lucas-Kanade optical flow (KLT) with forward-backward verification
- Parameters optimized for each dataset type
- Maintains a quality threshold to ensure reliable tracking

### Pose Estimation
- Uses P3P algorithm with RANSAC for outlier rejection
- Filters correspondences based on reprojection error
- Maintains motion consistency using previous pose when estimation fails

### Landmark Management
- Triangulates new landmarks when parallax angle exceeds threshold
- Verifies depth and reprojection error to ensure quality
- Maintains persistent landmark IDs across frames for bundle adjustment

### Visualization
- Current frame with tracked features and candidates
- Recent trajectory (last 20 frames) with visible landmarks
- Feature count history
- Full trajectory overview

## Results

The pipeline has been tested on three datasets:
- **KITTI** dataset: Outdoor driving sequences with large translations
- **Malaga** dataset: Urban environment with various motion patterns
- **Parking** dataset: More complex motion with significant rotations

Performance metrics:
- Tracking success rate: 85-95% on most sequences
- Pose estimation accuracy: Local consistency maintained well
- Processing speed: about 5 frames per second (depending on parameters)

## Performance Optimization

Several optimizations have been implemented to improve performance:
1. KD-tree for efficient landmark association
2. Selective keyframe processing for bundle adjustment
3. Adaptive feature detection based on tracking quality
4. Parallel processing for feature extraction and matching

## Troubleshooting

### Common Issues
- **Poor initialization**: Try different initial frames with more distinct motion
- **Tracking failures**: Adjust KLT parameters or reduce forward-backward threshold
- **Drift in rotation**: Increase keyframe frequency and bundle adjustment frequency
- **Scale drift**: Implement absolute scale recovery if ground truth is available, as by definition VO has scale ambiguity

### Parameter Tuning
The most important parameters to tune are:
- `forward_backward_threshold`: Controls keypoint tracking quality (higher = more keypoints, potentially more noise)
- `alpha_threshold`: Minimum parallax angle for triangulation (lower = more landmarks, potentially less accurate)
- `max_reprojection_error`: Maximum allowed reprojection error (higher = more landmarks, potentially more outliers)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The project structure is based on the assignment from the University of Zurich's Robotics and Perception Group.
- Datasets from KITTI, Malaga, and the Parking sequences.