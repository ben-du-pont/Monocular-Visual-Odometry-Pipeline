import cv2
import numpy as np
import time
import os
from copy import deepcopy
from feature_manager import FeatureManager, KeyPoint, Landmark
from pose_estimator import PoseEstimator
from triangulator import Triangulator
from bundle_adjuster import BundleAdjuster
from visualizer import Visualizer

class VisualOdometry:
    """
    A monocular visual odometry pipeline following the approach from:
    'Mini-project: A visual odometry pipeline!'
    """
    def __init__(self, dataset_loader, params=None):
        """
        Initialize the visual odometry pipeline.
        
        Args:
            dataset_loader: Dataset loader object
            params (dict, optional): Pipeline parameters
        """
        self.loader = dataset_loader
        self.K = self.loader.get_calibration()
        print(f"Camera calibration matrix:\n{self.K}")
        # Validate camera intrinsics
        assert self.K is not None and self.K.shape == (3, 3), "Invalid camera intrinsics"
        
        # Initialize default parameters (based on task description)
        self.params = {
            "initial_frames": (0, 3),            # Initial frames for bootstrapping
            "min_features": 100,                 # Minimum number of features to track
            "min_candidates": 500,               # Minimum number of candidate keypoints
            "max_features": 2000,                # Maximum number of features to detect
            "max_reprojection_error": 502.0,       # Maximum reprojection error
            "min_track_length": 3,               # Minimum track length for triangulation
            "min_bearing_angle": 1.0,            # Minimum bearing angle for triangulation (degrees)
            "forward_backward_threshold": 1.0,   # Maximum bidirectional error for tracking
            "use_bundle_adjustment": True,       # Whether to use bundle adjustment
            "ba_frequency": 10,                  # Bundle adjustment frequency (frames)
            "ba_window_size": 5,                 # Bundle adjustment window size
            "use_keyframes": True,               # Whether to use keyframes
            "keyframe_min_interval": 5,          # Minimum frames between keyframes
            "keyframe_max_interval": 20,         # Maximum frames between keyframes
            "keyframe_rotation_threshold": 5.0,  # Minimum rotation for new keyframe (degrees)
            "keyframe_translation_threshold": 0.5, # Minimum translation for new keyframe
            "visualize": True,                   # Whether to visualize results
        }
        
        # Override default parameters if provided
        if params:
            for key, value in params.items():
                self.params[key] = value
                
        # Initialize components
        self.feature_manager = FeatureManager(
            max_features=self.params["max_features"],
            quality_level=0.01,
            min_distance=7
        )
        
        self.pose_estimator = PoseEstimator()
        self.triangulator = Triangulator()
        
        if self.params["use_bundle_adjustment"]:
            self.bundle_adjuster = BundleAdjuster()
            
        if self.params["visualize"]:
            self.visualizer = Visualizer()
            
        # State variables
        self.frame_idx = 0
        self.poses = []
        self.keyframes = []
        self.landmarks = {}
        self.tracked_keypoints = []
        self.candidate_keypoints = []
        self.next_landmark_id = 0
        self.observations = {}  # (frame_idx, landmark_id) -> observation
        
        # Statistics
        self.processing_times = []
        self.tracking_rates = []
        self.inlier_ratios = []
        
    def initialize(self):
        """
        Initialize the VO pipeline using two frames with sufficient baseline.
        This follows the initialization module described in section 3 of the task.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        print("Initializing VO pipeline...")
        
        # Get the initial frames
        frame1_idx, frame2_idx = self.params["initial_frames"]
        frame1 = self.loader.get_image(frame1_idx)
        frame2 = self.loader.get_image(frame2_idx)
        
        if frame1 is None or frame2 is None:
            print("Error loading initial frames")
            return False
            
        # Detect keypoints in the first frame
        keypoints1 = self.feature_manager.detect_keypoints(frame1)
        
        if not keypoints1:
            print("No keypoints detected in the first frame")
            return False
            
        print(f"Detected {len(keypoints1)} keypoints in the first frame")
        
        # If using intermediate frames, track through them
        # This is mentioned in the task as an option for establishing correspondences
        intermediate_keypoints = keypoints1
        intermediate_frame = frame1
        
        for idx in range(frame1_idx + 1, frame2_idx):
            intermediate_frame_next = self.loader.get_image(idx)
            
            if intermediate_frame_next is None:
                continue
                
            # Track keypoints using KLT as recommended in section 4.1
            intermediate_keypoints, intermediate_indices = self.feature_manager.track_keypoints(
                intermediate_frame, intermediate_frame_next, intermediate_keypoints,
                max_error=self.params["forward_backward_threshold"]
            )
            
            print(f"Tracked {len(intermediate_keypoints)} keypoints to frame {idx}")
            
            intermediate_frame = intermediate_frame_next
            
        # Track to the final initialization frame
        keypoints2, keypoints2_indices = self.feature_manager.track_keypoints(
            intermediate_frame, frame2, intermediate_keypoints,
            max_error=self.params["forward_backward_threshold"]
        )
        
        print(f"Tracked {len(keypoints2)} keypoints to the second initialization frame")
        
        # Ensure we have enough keypoints (minimum 5 required for essential matrix)
        if len(keypoints2) < 5:
            print("Not enough keypoints tracked between initialization frames")
            return False
        
        intermediate_keypoints_tracked = [intermediate_keypoints[i] for i in keypoints2_indices]

        try:
            # Estimate relative pose using 2D-2D correspondences (essential matrix)
            # As described in section 3 of the task
            success, T_1_2, inliers = self.pose_estimator.estimate_pose_2d2d(
                self.K, intermediate_keypoints_tracked, keypoints2,
                ransac_threshold=1.0, confidence=0.999
            )
            
            if not success:
                print("Failed to estimate relative pose between initialization frames")
                return False
            
            print(f"Estimated relative pose with {len(inliers)} inliers")
            
            # Filter keypoints using inliers
            keypoints1_inliers = [intermediate_keypoints[i] for i in inliers]
            keypoints2_inliers = [keypoints2[i] for i in inliers]
            
            # Set initial poses following the task description:
            # First camera is at the origin, second camera at the relative pose
            T_w_1 = np.eye(4)  # First camera is at the origin
            T_w_2 = T_w_1 @ T_1_2  # Second camera is at the relative pose from first
            print("Relative translation magnitude:", np.linalg.norm(T_1_2[:3, 3]))

            cam1_center = -np.linalg.inv(T_w_1[:3, :3]) @ T_w_1[:3, 3]
            cam2_center = -np.linalg.inv(T_w_2[:3, :3]) @ T_w_2[:3, 3]
            baseline = np.linalg.norm(cam2_center - cam1_center)
            print(f"Camera 1 center: {cam1_center}")
            print(f"Camera 2 center: {cam2_center}")
            print(f"Baseline: {baseline}")

            # Triangulate landmarks as described in section 3 of the task
            landmarks, keypoints1_filtered, keypoints2_filtered = self.triangulator.triangulate_with_bearing_angle(
                self.K, T_w_1, T_w_2, keypoints1_inliers, keypoints2_inliers,
                min_angle_deg=self.params["min_bearing_angle"], 
                max_error=self.params["max_reprojection_error"]
            )
            
            print(f"Triangulated {len(landmarks)} landmarks")
            
            if len(landmarks) < 10:
                print("Not enough landmarks triangulated")
                return False
            
            # Initialize the state
            self.frame_idx = frame2_idx
            self.poses = [T_w_1, T_w_2]
            self.keyframes = [(frame1_idx, T_w_1), (frame2_idx, T_w_2)]
            
            # Store landmarks and their observations
            for i, landmark in enumerate(landmarks):
                self.landmarks[self.next_landmark_id] = landmark
                
                # Store observations
                self.observations[(frame1_idx, self.next_landmark_id)] = keypoints1_filtered[i].uv
                self.observations[(frame2_idx, self.next_landmark_id)] = keypoints2_filtered[i].uv
                
                # Set up tracked keypoints
                keypoint = deepcopy(keypoints2_filtered[i])
                keypoint.landmark_id = self.next_landmark_id
                self.tracked_keypoints.append(keypoint)
                
                self.next_landmark_id += 1
            
            # Detect new candidate keypoints
            mask = self.feature_manager.create_candidates_mask(
                frame2, self.tracked_keypoints, min_distance=7
            )
            
            candidate_keypoints = self.feature_manager.detect_keypoints(frame2, mask)
            print(f"Detected {len(candidate_keypoints)} new candidate keypoints")
            
            self.candidate_keypoints = candidate_keypoints
            
            # Visualize initialization results
            if self.params["visualize"]:
                self.visualizer.update(
                    frame2, T_w_2, list(self.landmarks.values()),
                    self.tracked_keypoints, self.candidate_keypoints, self.frame_idx
                )
            
            print("Initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"Initialization failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def process_frame(self, frame):
        """
        Process a new frame and update the VO state.
        This implements the continuous operation module described in section 4 of the task.
        
        Args:
            frame: New image frame
            
        Returns:
            numpy.ndarray: Current camera pose (4x4)
        """
        # Validate inputs
        assert frame is not None, "Frame cannot be None"
        
        if not self.tracked_keypoints or not self.poses:
            print("VO not initialized. Call initialize() first.")
            return np.eye(4)
            
        # Get the previous frame and pose
        prev_frame = self.loader.get_image(self.frame_idx)
        prev_pose = self.poses[-1]
        
        # Start timing
        start_time = time.time()
        
        # 1. Associate keypoints to existing landmarks (section 4.1)
        # Track keypoints using KLT as recommended
        tracked_keypoints, tracked_keypoints_indices = self.feature_manager.track_keypoints(
            prev_frame, frame, self.tracked_keypoints,
            max_error=self.params["forward_backward_threshold"]
        )
        
        tracking_ratio = len(tracked_keypoints) / len(self.tracked_keypoints) if self.tracked_keypoints else 0
        print(f"Tracked {len(tracked_keypoints)}/{len(self.tracked_keypoints)} keypoints ({tracking_ratio:.2f})")
        self.tracking_rates.append(tracking_ratio)
        
        # Extract landmarks corresponding to tracked keypoints
        landmarks = []
        keypoints = []
        landmark_ids = []
        
        for kp in tracked_keypoints:
            if hasattr(kp, 'landmark_id') and kp.landmark_id in self.landmarks:
                landmarks.append(self.landmarks[kp.landmark_id])
                keypoints.append(kp)
                landmark_ids.append(kp.landmark_id)
        
        if len(landmarks) < 4:
            print(f"Warning: Too few landmarks for PnP ({len(landmarks)})")
            # Use motion model if not enough landmarks
            pose = self.pose_estimator.apply_motion_model(self.poses)
            inliers = list(range(len(landmarks)))
        else:
            # 2. Estimate the current pose (section 4.2)
            # Use P3P with RANSAC as recommended for pose estimation
            success, pose, inliers = self.pose_estimator.estimate_pose_3d2d(
                self.K, landmarks, keypoints,
                pnp_method=cv2.SOLVEPNP_P3P,
                ransac=True,
                reprojection_error=self.params["max_reprojection_error"]
            )
            
            inlier_ratio = len(inliers) / len(landmarks) if landmarks else 0
            print(f"Pose estimation: {len(inliers)}/{len(landmarks)} inliers ({inlier_ratio:.2f})")
            self.inlier_ratios.append(inlier_ratio)
            
            # Verify pose plausibility
            pose_is_plausible = self.pose_estimator.verify_pose(
                prev_pose, pose, max_rotation_deg=30.0, max_translation=2.0
            )
            
            if not success or not pose_is_plausible:
                print("Warning: Pose estimation failed or implausible. Using motion model.")
                pose = self.pose_estimator.apply_motion_model(self.poses)
                inliers = list(range(len(landmarks)))
                
        # Update tracked keypoints and landmarks based on inliers
        self.tracked_keypoints = [keypoints[i] for i in inliers] if inliers else []
        
        # Store observations for inliers
        self.frame_idx += 1
        for i, idx in enumerate(inliers):
            if idx < len(landmark_ids):
                landmark_id = landmark_ids[idx]
                self.observations[(self.frame_idx, landmark_id)] = keypoints[idx].uv
            
        # Update poses
        self.poses.append(pose)
        
        # Check if this should be a keyframe
        is_keyframe = self._is_keyframe(pose)
        
        # 3. Triangulate new landmarks (section 4.3)
        # Track candidate keypoints
        candidate_keypoints, candidate_keypoint_indices = self.feature_manager.track_keypoints(
            prev_frame, frame, self.candidate_keypoints,
            max_error=self.params["forward_backward_threshold"]
        )
        
        print(f"Tracked {len(candidate_keypoints)}/{len(self.candidate_keypoints)} candidate keypoints")
        
        # Triangulate new landmarks from candidates if this is a keyframe
        if is_keyframe:
            self._add_keyframe(pose)
            
            # We need at least one previous keyframe
            if len(self.keyframes) > 1:
                prev_keyframe_idx, prev_keyframe_pose = self.keyframes[-2]
                
                # Find candidates with sufficient track length
                candidates_to_triangulate = []
                for kp in candidate_keypoints:
                    if kp.t_total >= self.params["min_track_length"]:
                        candidates_to_triangulate.append(kp)
                        
                print(f"Triangulating {len(candidates_to_triangulate)} candidate keypoints")
                
                # Triangulate using the current and previous keyframe
                if candidates_to_triangulate:
                    # We need to collect matching observations in the previous keyframe
                    prev_keyframe_img = self.loader.get_image(prev_keyframe_idx)
                    keypoints_prev = []
                    keypoints_curr = []
                    
                    for kp in candidates_to_triangulate:
                        # Find the observation at the previous keyframe
                        # For simplicity, we'll just use the first observation
                        if len(kp.uv_history) >= kp.t_total:
                            prev_obs = kp.uv_history[0]
                            
                            # Create keypoint objects for triangulation
                            kp_prev = KeyPoint(prev_obs, t_first=prev_keyframe_idx)
                            keypoints_prev.append(kp_prev)
                            keypoints_curr.append(kp)
                    
                    # Triangulate new landmarks as described in section 4.3
                    # Using bearing angle threshold to ensure good triangulation
                    if keypoints_prev and keypoints_curr:
                        new_landmarks, _, new_keypoints = self.triangulator.triangulate_with_bearing_angle(
                            self.K, prev_keyframe_pose, pose,
                            keypoints_prev, keypoints_curr,
                            min_angle_deg=self.params["min_bearing_angle"],
                            max_error=self.params["max_reprojection_error"]
                        )
                        
                        print(f"Successfully triangulated {len(new_landmarks)} new landmarks")
                        
                        # Add new landmarks and update tracked keypoints
                        for i, landmark in enumerate(new_landmarks):
                            self.landmarks[self.next_landmark_id] = landmark
                            
                            # Update keypoint
                            kp = new_keypoints[i]
                            kp.landmark_id = self.next_landmark_id
                            
                            # Store observations
                            self.observations[(prev_keyframe_idx, self.next_landmark_id)] = keypoints_prev[i].uv
                            self.observations[(self.frame_idx, self.next_landmark_id)] = kp.uv
                            
                            # Add to tracked keypoints
                            self.tracked_keypoints.append(kp)
                            
                            # Remove from candidates
                            if kp in candidate_keypoints:
                                candidate_keypoints.remove(kp)
                                
                            self.next_landmark_id += 1
                
        # Update candidate keypoints
        self.candidate_keypoints = candidate_keypoints
        
        # Detect new candidate keypoints if needed
        if len(self.tracked_keypoints) < self.params["min_features"] or \
           len(self.candidate_keypoints) < self.params["min_candidates"]:
            
            # Create mask to avoid existing features
            mask = self.feature_manager.create_candidates_mask(
                frame,
                self.tracked_keypoints + self.candidate_keypoints,
                min_distance=7
            )
            
            # Detect new keypoints
            new_candidates = self.feature_manager.detect_keypoints(frame, mask)
            print(f"Detected {len(new_candidates)} new candidate keypoints")
            
            # Add to candidate keypoints
            self.candidate_keypoints.extend(new_candidates)
            
        # Perform bundle adjustment if enabled (bonus feature in task description)
        if self.params["use_bundle_adjustment"] and \
           self.frame_idx % self.params["ba_frequency"] == 0 and \
           len(self.poses) > self.params["ba_window_size"]:
            
            print("Performing bundle adjustment...")
            updated_poses, updated_landmarks = self.bundle_adjuster.local_bundle_adjustment(
                self.poses, self.landmarks, self.observations,
                self.K, window_size=self.params["ba_window_size"]
            )
            
            self.poses = updated_poses
            self.landmarks = updated_landmarks
            
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        print(f"Frame {self.frame_idx} processed in {processing_time:.3f}s")
        
        # Visualize results
        if self.params["visualize"]:
            self.visualizer.update(
                frame, pose, list(self.landmarks.values()),
                self.tracked_keypoints, self.candidate_keypoints, self.frame_idx
            )
            
        return pose
        
    def _is_keyframe(self, pose):
        """
        Determine if the current frame should be a keyframe.
        
        Args:
            pose (numpy.ndarray): Current camera pose (4x4)
            
        Returns:
            bool: True if the frame should be a keyframe, False otherwise
        """
        if not self.params["use_keyframes"]:
            return False
            
        if not self.keyframes:
            return True
            
        # Get the last keyframe
        last_keyframe_idx, last_keyframe_pose = self.keyframes[-1]
        
        # Check if enough frames have passed since the last keyframe
        if self.frame_idx - last_keyframe_idx < self.params["keyframe_min_interval"]:
            return False
            
        # Check if too many frames have passed since the last keyframe
        if self.frame_idx - last_keyframe_idx >= self.params["keyframe_max_interval"]:
            return True
            
        # Check if there's been enough motion since the last keyframe
        try:
            relative_pose = np.linalg.inv(last_keyframe_pose) @ pose
            
            # Extract rotation and translation
            from scipy.spatial.transform import Rotation
            R = relative_pose[:3, :3]
            t = relative_pose[:3, 3]
            
            # Compute rotation angle
            r = Rotation.from_matrix(R)
            angles_deg = np.abs(r.as_euler('xyz', degrees=True))
            max_angle_deg = np.max(angles_deg)
            
            # Compute translation magnitude
            translation_mag = np.linalg.norm(t)
            
            # Check if there's been enough motion
            rotation_threshold_exceeded = max_angle_deg >= self.params["keyframe_rotation_threshold"]
            translation_threshold_exceeded = translation_mag >= self.params["keyframe_translation_threshold"]
            
            return rotation_threshold_exceeded or translation_threshold_exceeded
        except Exception as e:
            print(f"Error in keyframe decision: {e}")
            # Default to false if there's an error
            return False
        
    def _add_keyframe(self, pose):
        """
        Add a new keyframe to the VO state.
        
        Args:
            pose (numpy.ndarray): Camera pose of the keyframe (4x4)
        """
        print(f"Adding keyframe at frame {self.frame_idx}")
        self.keyframes.append((self.frame_idx, pose))
        
    def run(self):
        """
        Run the full VO pipeline on the dataset.
        """
        print("Running VO pipeline...")
        
        # Initialize the pipeline
        if not self.initialize():
            print("Initialization failed. Exiting.")
            return False
            
        # Process all remaining frames
        for frame_idx in range(self.frame_idx + 1, self.loader.get_num_images()):
            frame = self.loader.get_image(frame_idx)
            
            if frame is None:
                print(f"Failed to load frame {frame_idx}. Skipping.")
                continue
                
            # Process the frame
            try:
                pose = self.process_frame(frame)
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                break
                
        # Print summary statistics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_tracking_rate = np.mean(self.tracking_rates) if self.tracking_rates else 0
        avg_inlier_ratio = np.mean(self.inlier_ratios) if self.inlier_ratios else 0
        
        print("\nVO Pipeline Summary:")
        print(f"  Processed {len(self.poses)} frames")
        print(f"  Average processing time: {avg_processing_time:.3f}s ({1/avg_processing_time:.2f} FPS)")
        print(f"  Average tracking rate: {avg_tracking_rate:.2f}")
        print(f"  Average inlier ratio: {avg_inlier_ratio:.2f}")
        print(f"  Final trajectory length: {len(self.poses)}")
        print(f"  Number of keyframes: {len(self.keyframes)}")
        print(f"  Number of landmarks: {len(self.landmarks)}")
        
        return True
        
    def save_results(self, output_dir):
        """
        Save the VO results to files.
        
        Args:
            output_dir (str): Output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save trajectory
        if self.params["visualize"]:
            # Save the final visualization
            self.visualizer.save_figure(os.path.join(output_dir, 'trajectory.png'))
            
            # Save the trajectory to a file
            self.visualizer.save_trajectory(os.path.join(output_dir, 'trajectory.txt'), self.poses)
            
        # Save statistics
        stats = {
            'processing_times': self.processing_times,
            'tracking_rates': self.tracking_rates,
            'inlier_ratios': self.inlier_ratios
        }
        
        import pickle
        with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
            pickle.dump(stats, f)
            
        print(f"Results saved to {output_dir}")