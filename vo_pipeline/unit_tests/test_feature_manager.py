import unittest
import numpy as np
import cv2
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import the modules
sys.path.append(str(Path(__file__).parent.parent))
from feature_manager import KeyPoint, Landmark, FeatureManager
from dataset_loader import DatasetLoader

# Global flag to control visualization
SHOW_VISUALIZATIONS = True  # Set to False to run tests without UI
SAVE_VISUALIZATIONS = True  # Save visualizations to disk instead of showing
VIZ_OUTPUT_DIR = "visualization_output"  # Directory to save visualizations

def create_output_dir():
    """Create output directory for visualizations if it doesn't exist."""
    if SAVE_VISUALIZATIONS:
        os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

def show_or_save_image(window_name, img):
    """Helper function to show or save an image based on visualization flags."""
    global SHOW_VISUALIZATIONS

    if SHOW_VISUALIZATIONS:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1000)  # Wait for 1 second
        if key == 27:  # ESC key
            cv2.destroyAllWindows()
            SHOW_VISUALIZATIONS = False
    
    if SAVE_VISUALIZATIONS:
        create_output_dir()
        # Replace any spaces with underscores and make lowercase
        safe_name = window_name.replace(" ", "_").lower()
        filename = os.path.join(VIZ_OUTPUT_DIR, f"{safe_name}_{int(time.time())}.png")
        cv2.imwrite(filename, img)
        print(f"Saved visualization to {filename}")



class TestKeyPoint(unittest.TestCase):
    """Test cases for the KeyPoint class."""
    
    def test_initialization(self):
        """Test initialization with different input formats."""
        # Test with 1D numpy array
        kp1 = KeyPoint(np.array([10.5, 20.5], dtype=np.float32))
        self.assertEqual(kp1.uv.shape, (2,))
        self.assertEqual(kp1.uv.dtype, np.float32)
        self.assertAlmostEqual(kp1.uv[0], 10.5)
        self.assertAlmostEqual(kp1.uv[1], 20.5)
        
        # Test with 2D array
        kp2 = KeyPoint(np.array([[30.5, 40.5]], dtype=np.float32))
        self.assertEqual(kp2.uv.shape, (2,))
        self.assertAlmostEqual(kp2.uv[0], 30.5)
        self.assertAlmostEqual(kp2.uv[1], 40.5)
        
        # Test with list
        kp3 = KeyPoint(np.array([50.5, 60.5], dtype=np.float32))
        self.assertEqual(kp3.uv.shape, (2,))
        self.assertAlmostEqual(kp3.uv[0], 50.5)
        self.assertAlmostEqual(kp3.uv[1], 60.5)
        
        # Test with too many dimensions - should only take first two elements
        kp4 = KeyPoint(np.array([70.5, 80.5, 90.5], dtype=np.float32))
        self.assertEqual(kp4.uv.shape, (2,))
        self.assertAlmostEqual(kp4.uv[0], 70.5)
        self.assertAlmostEqual(kp4.uv[1], 80.5)
        
        # Test first frame tracking and history
        kp5 = KeyPoint(np.array([10.5, 20.5], dtype=np.float32), t_first=5)
        self.assertEqual(kp5.t_first, 5)
        self.assertEqual(len(kp5.uv_history), 1)
        np.testing.assert_array_almost_equal(kp5.uv_history[0], np.array([10.5, 20.5]))
    
    def test_update(self):
        """Test updating keypoint position."""
        kp = KeyPoint(np.array([10.5, 20.5], dtype=np.float32))
        kp.update(np.array([15.5, 25.5], dtype=np.float32))
        
        # Check that uv is updated
        self.assertAlmostEqual(kp.uv[0], 15.5)
        self.assertAlmostEqual(kp.uv[1], 25.5)
        
        # Check that history is updated correctly
        self.assertEqual(len(kp.uv_history), 2)
        np.testing.assert_array_almost_equal(kp.uv_history[0], np.array([10.5, 20.5]))
        np.testing.assert_array_almost_equal(kp.uv_history[1], np.array([15.5, 25.5]))
        
        # Check that t_total is incremented
        self.assertEqual(kp.t_total, 2)
        
        # Test multiple updates
        kp.update(np.array([20.5, 30.5]))
        self.assertEqual(len(kp.uv_history), 3)
        self.assertEqual(kp.t_total, 3)


class TestLandmark(unittest.TestCase):
    """Test cases for the Landmark class."""
    
    def test_initialization(self):
        """Test initialization with different input formats."""
        # Test with 1D numpy array
        lm1 = Landmark(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertEqual(lm1.p.shape, (3,))
        self.assertEqual(lm1.p.dtype, np.float32)
        self.assertAlmostEqual(lm1.p[0], 1.0)
        self.assertAlmostEqual(lm1.p[1], 2.0)
        self.assertAlmostEqual(lm1.p[2], 3.0)
        
        # Test with 2D array
        lm2 = Landmark(np.array([[4.0, 5.0, 6.0]], dtype=np.float32))
        self.assertEqual(lm2.p.shape, (3,))
        self.assertAlmostEqual(lm2.p[0], 4.0)
        self.assertAlmostEqual(lm2.p[1], 5.0)
        self.assertAlmostEqual(lm2.p[2], 6.0)
        
        # Test with list
        lm3 = Landmark(np.array([7.0, 8.0, 9.0], dtype=np.float32))
        self.assertEqual(lm3.p.shape, (3,))
        self.assertAlmostEqual(lm3.p[0], 7.0)
        self.assertAlmostEqual(lm3.p[1], 8.0)
        self.assertAlmostEqual(lm3.p[2], 9.0)
        
        # Test with too many dimensions - should only take first three elements
        lm4 = Landmark(np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32))
        self.assertEqual(lm4.p.shape, (3,))
        self.assertAlmostEqual(lm4.p[0], 10.0)
        self.assertAlmostEqual(lm4.p[1], 11.0)
        self.assertAlmostEqual(lm4.p[2], 12.0)
        
        # Test first frame tracking
        lm5 = Landmark(np.array([1.0, 2.0, 3.0], dtype=np.float32), t_first=5)
        self.assertEqual(lm5.t_first, 5)
        self.assertEqual(lm5.observed_count, 1)
    
    def test_update(self):
        """Test updating landmark position."""
        lm = Landmark(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        lm.update(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        
        # Check that p is updated
        self.assertAlmostEqual(lm.p[0], 4.0)
        self.assertAlmostEqual(lm.p[1], 5.0)
        self.assertAlmostEqual(lm.p[2], 6.0)
        
        # Check that observed_count is incremented
        self.assertEqual(lm.observed_count, 2)


class TestFeatureManager(unittest.TestCase):
    """Test cases for the FeatureManager class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with sample images."""
        # Default dataset paths based on the main script structure
        DEFAULT_PATHS = {
            'kitti': os.path.abspath("datasets/kitti/05"),
            'malaga': os.path.abspath("datasets/malaga"),
            'parking': os.path.abspath("datasets/parking")
        }
        
        # Try additional common locations
        base_dir = Path(__file__).parent.parent
        additional_paths = [
            os.path.join(base_dir, "datasets", "kitti", "05"),
            os.path.join(base_dir, "data", "kitti", "05"),
            os.path.join(base_dir, "datasets", "kitti"),
            os.path.join(base_dir, "data", "kitti"),
            "datasets/kitti",
            "data/kitti"
        ]
        
        # Combine all possible paths
        all_paths = [DEFAULT_PATHS['kitti']] + additional_paths
        
        # Print available paths for debugging
        print("Searching for KITTI dataset in:")
        for path in all_paths:
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  - {path} {exists}")
        
        # Try to load the dataset
        cls.dataset = None
        for path in all_paths:
            if os.path.exists(path):
                try:
                    print(f"Attempting to load dataset from: {path}")
                    cls.dataset = DatasetLoader(path, "kitti")
                    print(f"Successfully loaded KITTI dataset from: {path}")
                    # Get a few images to verify
                    test_img = cls.dataset.get_image(0)
                    if test_img is not None:
                        print(f"Test image loaded successfully: {test_img.shape}")
                        break
                    else:
                        print("Test image load failed, trying next path")
                except Exception as e:
                    print(f"Failed to load dataset from {path}: {e}")
        
        # Create synthetic test images
        cls.test_images = []
        base_img = np.zeros((480, 640), dtype=np.uint8)
        
        # Image 1: Random noise + circles
        img1 = base_img.copy()
        img1 = np.random.randint(0, 50, size=img1.shape, dtype=np.uint8)
        for i in range(10):
            x = np.random.randint(50, 590)
            y = np.random.randint(50, 430)
            cv2.circle(img1, (x, y), 10, 255, -1)
        cls.test_images.append(img1)
        
        # Image 2: Slightly shifted version of image 1
        img2 = np.zeros_like(img1)
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
        cls.test_images.append(img2)
        
        # Image 3: More shifted version
        img3 = np.zeros_like(img1)
        M = np.float32([[1, 0, 10], [0, 1, 7]])
        img3 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
        cls.test_images.append(img3)
        
        if cls.dataset is None:
            print("No KITTI dataset found, will use only synthetic test images")
        
        # Create output directory for visualizations
        create_output_dir()

    
    def get_test_image(self, index=0):
        """Get a test image, either from the dataset or synthetic."""
        if hasattr(self.__class__, 'dataset') and self.__class__.dataset is not None:
            img = self.__class__.dataset.get_image(index)
            if img is not None:
                return img
        return self.__class__.test_images[min(index, len(self.__class__.test_images)-1)]
    
    def test_detect_keypoints(self):
        """Test keypoint detection."""
        feature_manager = FeatureManager(max_features=500)
        img = self.get_test_image(0)
        
        # Test with SIFT
        feature_manager.use_sift = True
        keypoints = feature_manager.detect_keypoints(img)
        
        # Check if keypoints were detected
        self.assertGreater(len(keypoints), 0, "No SIFT keypoints detected")
        
        # Check if each keypoint has the correct format
        for kp in keypoints:
            self.assertEqual(kp.uv.shape, (2,))
            self.assertEqual(kp.uv.dtype, np.float32)
            if kp.des is not None:
                self.assertEqual(len(kp.des), 128)  # SIFT descriptor length
        
        # Test with goodFeaturesToTrack
        feature_manager.use_sift = False
        keypoints = feature_manager.detect_keypoints(img)
        
        # Check if keypoints were detected
        self.assertGreater(len(keypoints), 0, "No goodFeaturesToTrack keypoints detected")
        
        # Check if each keypoint has the correct format
        for kp in keypoints:
            self.assertEqual(kp.uv.shape, (2,))
            self.assertEqual(kp.uv.dtype, np.float32)
    
    def test_track_keypoints(self):
        """Test keypoint tracking between frames."""
        feature_manager = FeatureManager(max_features=500)
        prev_img = self.get_test_image(0)
        curr_img = self.get_test_image(1)
        
        # Detect keypoints in first frame
        keypoints = feature_manager.detect_keypoints(prev_img)
        self.assertGreater(len(keypoints), 0, "No keypoints detected for tracking test")
        
        # Track to second frame
        tracked_keypoints, tracked_indices = feature_manager.track_keypoints(prev_img, curr_img, keypoints)
        
        # Check if some keypoints were successfully tracked
        # For synthetic images we expect high tracking success; for real images it may vary
        tracked_ratio = len(tracked_keypoints) / len(keypoints)
        min_expected_ratio = 0.3  # Lower threshold for real-world images
        
        self.assertGreater(tracked_ratio, min_expected_ratio, 
                          f"Too few keypoints tracked ({tracked_ratio:.2%}). Expected >{min_expected_ratio:.0%}")
        
        # Check if tracked keypoints maintained their format
        for kp in tracked_keypoints:
            self.assertEqual(kp.uv.shape, (2,))
            self.assertEqual(kp.uv.dtype, np.float32)
            self.assertEqual(kp.t_total, 2)  # One update means t_total should be 2
            self.assertEqual(len(kp.uv_history), 2)  # Should have two positions in history
    
    def test_create_candidates_mask(self):
        """Test creation of mask for candidate keypoints."""
        feature_manager = FeatureManager()
        img = self.get_test_image(0)
        
        # Create some keypoints
        keypoints = [KeyPoint(np.array([100, 100], dtype=np.float32)), 
                     KeyPoint(np.array([200, 200], dtype=np.float32))]
        
        # Create mask
        mask = feature_manager.create_candidates_mask(img, keypoints, min_distance=20)
        
        # Check mask shape
        self.assertEqual(mask.shape, img.shape)
        
        # Check mask values around keypoints
        self.assertEqual(mask[100, 100], 0, "Keypoint center should be masked")
        self.assertEqual(mask[80, 100], 0, "Area around keypoint should be masked")
        self.assertEqual(mask[100, 80], 0, "Area around keypoint should be masked")
        
        # Only check far areas if within image bounds
        if img.shape[0] > 300 and img.shape[1] > 300:
            self.assertEqual(mask[300, 300], 255, "Far from keypoints should not be masked")
    
    def test_match_keypoints_with_sift(self):
        """Test keypoint matching between frames using SIFT descriptors."""
        feature_manager = FeatureManager(max_features=500)
        feature_manager.use_sift = True
        
        img1 = self.get_test_image(0)
        img2 = self.get_test_image(1)
        
        # Detect keypoints in both frames
        keypoints1 = feature_manager.detect_keypoints(img1)
        keypoints2 = feature_manager.detect_keypoints(img2)
        
        # Skip test if not enough keypoints with descriptors
        if not all([kp.des is not None for kp in keypoints1]) or not all([kp.des is not None for kp in keypoints2]):
            self.skipTest("Keypoints don't have descriptors for matching test")
        
        # Skip test if not enough keypoints
        if len(keypoints1) < 10 or len(keypoints2) < 10:
            self.skipTest("Not enough keypoints detected for matching test")
        
        # Match keypoints
        matches = feature_manager.match_keypoints(keypoints1, keypoints2)
        
        # Check if some matches were found (may be zero for very different frames)
        found_matches = len(matches) > 0
        
        if not found_matches:
            print("Warning: No matches found between frames in match_keypoints test")
        else:
            # Verify match format
            for match in matches:
                self.assertEqual(len(match), 2, "Match should be a tuple of (index1, index2)")
                self.assertLess(match[0], len(keypoints1), "Match index out of range")
                self.assertLess(match[1], len(keypoints2), "Match index out of range")
    
    def test_end_to_end_tracking(self):
        """Test the entire tracking pipeline over multiple frames."""
        feature_manager = FeatureManager(max_features=300)
        feature_manager.use_sift = False  # Use KLT tracking
        
        # Get sequential frames (either from dataset or synthetic)
        frames = []
        for i in range(3):  # Get 3 consecutive frames
            frames.append(self.get_test_image(i))
        
        # Detect keypoints in first frame
        keypoints = feature_manager.detect_keypoints(frames[0])
        self.assertGreater(len(keypoints), 0, "No keypoints detected in first frame")
        
        # Track through frames
        frame_keypoints = [keypoints]
        for i in range(1, len(frames)):
            tracked_kps, tracked_indices = feature_manager.track_keypoints(frames[i-1], frames[i], frame_keypoints[i-1])
            frame_keypoints.append(tracked_kps)
            
            # Check that we retained a reasonable number of keypoints
            retention_ratio = len(tracked_kps) / len(frame_keypoints[i-1])
            min_expected_ratio = 0.5  # Lower threshold for real-world images
            
            self.assertGreater(retention_ratio, min_expected_ratio, 
                              f"Too many keypoints lost in tracking to frame {i}")
            
            # Check that keypoints have updated history
            for kp in tracked_kps:
                self.assertEqual(kp.t_total, i+1, f"t_total should be {i+1}")
                self.assertEqual(len(kp.uv_history), i+1, f"uv_history length should be {i+1}")

    def test_detect_keypoints_with_visualization(self):
        """Test keypoint detection with visualization."""
        feature_manager = FeatureManager(max_features=500)
        img = self.get_test_image(0)
        
        # Test with SIFT
        feature_manager.use_sift = True
        keypoints = feature_manager.detect_keypoints(img)
        
        # Check if keypoints were detected
        self.assertGreater(len(keypoints), 0, "No SIFT keypoints detected")
        
        # Visualize keypoints
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Convert to BGR if it's grayscale
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
                
            # Draw keypoints
            for kp in keypoints:
                cv2.circle(img_color, 
                          (int(kp.uv[0]), int(kp.uv[1])), 
                          5, (0, 255, 0), 2)
            
            # Add text with keypoint count
            cv2.putText(img_color, f"SIFT Keypoints: {len(keypoints)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
            
            show_or_save_image("SIFT Keypoints", img_color)
        
        # Test with goodFeaturesToTrack
        feature_manager.use_sift = False
        keypoints = feature_manager.detect_keypoints(img)
        
        # Check if keypoints were detected
        self.assertGreater(len(keypoints), 0, "No goodFeaturesToTrack keypoints detected")
        
        # Visualize keypoints
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Convert to BGR if it's grayscale
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
                
            # Draw keypoints
            for kp in keypoints:
                cv2.circle(img_color, 
                          (int(kp.uv[0]), int(kp.uv[1])), 
                          5, (0, 0, 255), 2)
            
            # Add text with keypoint count
            cv2.putText(img_color, f"Harris Keypoints: {len(keypoints)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2)
            
            show_or_save_image("Harris Keypoints", img_color)
    
    def test_track_keypoints_with_visualization(self):
        """Test keypoint tracking between frames with visualization."""
        feature_manager = FeatureManager(max_features=500)
        prev_img = self.get_test_image(0)
        curr_img = self.get_test_image(1)
        
        # Detect keypoints in first frame
        keypoints = feature_manager.detect_keypoints(prev_img)
        self.assertGreater(len(keypoints), 0, "No keypoints detected for tracking test")
        
        # Track to second frame
        tracked_keypoints, tracked_indices = feature_manager.track_keypoints(prev_img, curr_img, keypoints)
        
        # Visualize tracking
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Create a visualization image
            feature_manager.visualize_tracks(
                prev_img, curr_img, 
                [keypoints[i] for i in tracked_indices], 
                tracked_keypoints,
                "Feature Tracking"
            )
            
            # Create manual visualization
            if len(prev_img.shape) == 2:
                prev_color = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)
                curr_color = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
            else:
                prev_color = prev_img.copy()
                curr_color = curr_img.copy()
            
            # Draw original keypoints in first frame
            for kp in keypoints:
                cv2.circle(prev_color, (int(kp.uv[0]), int(kp.uv[1])), 
                          5, (255, 0, 0), 2)
            
            # Draw tracked keypoints in second frame
            for kp in tracked_keypoints:
                cv2.circle(curr_color, (int(kp.uv[0]), int(kp.uv[1])), 
                          5, (0, 255, 0), 2)
            
            # Create side-by-side display
            h, w = prev_color.shape[:2]
            vis = np.zeros((h, w*2, 3), dtype=np.uint8)
            vis[:, :w] = prev_color
            vis[:, w:] = curr_color
            
            # Add text with tracking info
            tracked_ratio = len(tracked_keypoints) / len(keypoints) * 100
            cv2.putText(vis, f"Original: {len(keypoints)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(vis, f"Tracked: {len(tracked_keypoints)} ({tracked_ratio:.1f}%)", 
                       (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw lines between tracked points
            for i, kp2 in enumerate(tracked_keypoints):
                kp1 = keypoints[tracked_indices[i]]
                pt1 = (int(kp1.uv[0]), int(kp1.uv[1]))
                pt2 = (int(kp2.uv[0]) + w, int(kp2.uv[1]))
                cv2.line(vis, pt1, pt2, (0, 255, 255), 1)
            
            show_or_save_image("Keypoint Tracking", vis)
    
    def test_tracking_sequence_with_visualization(self):
        """Test tracking over a sequence of frames with visualization."""
        feature_manager = FeatureManager(max_features=300)
        feature_manager.use_sift = False  # Use KLT tracking
        
        # Get more frames if available
        num_frames = 5  # Increase this for more frames in the sequence
        frames = []
        
        # Try to get real dataset frames
        if hasattr(self.__class__, 'dataset') and self.__class__.dataset is not None:
            for i in range(num_frames):
                img = self.__class__.dataset.get_image(i)
                if img is not None:
                    frames.append(img)
                else:
                    break
        
        # Fall back to synthetic frames if needed
        if not frames:
            for i in range(min(num_frames, len(self.__class__.test_images))):
                frames.append(self.__class__.test_images[i])
        
        # Initialize video writer if saving
        if SAVE_VISUALIZATIONS:
            create_output_dir()
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = os.path.join(VIZ_OUTPUT_DIR, f"tracking_sequence_{int(time.time())}.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (w*2, h))
            print(f"Saving tracking sequence to {video_path}")
        else:
            video_writer = None
        
        # Detect keypoints in first frame
        keypoints = feature_manager.detect_keypoints(frames[0])
        prev_frame = frames[0]
        
        # Draw keypoints on first frame
        if len(prev_frame.shape) == 2:
            vis_frame = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_frame = prev_frame.copy()
            
        for kp in keypoints:
            cv2.circle(vis_frame, (int(kp.uv[0]), int(kp.uv[1])), 
                      3, (0, 255, 0), -1)
            
        cv2.putText(vis_frame, f"Frame 0: {len(keypoints)} keypoints", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Create full visualization with empty right side for first frame
        h, w = vis_frame.shape[:2]
        full_vis = np.zeros((h, w*2, 3), dtype=np.uint8)
        full_vis[:, :w] = vis_frame
        
        if SHOW_VISUALIZATIONS:
            cv2.imshow("Tracking Sequence", full_vis)
            cv2.waitKey(500)  # Wait for 500ms
            
        if video_writer:
            video_writer.write(full_vis)
        
        # Track through remaining frames
        frame_keypoints = [keypoints]
        
        for i in range(1, len(frames)):
            curr_frame = frames[i]
            tracked_kps, tracked_indices = feature_manager.track_keypoints(prev_frame, curr_frame, frame_keypoints[i-1])
            frame_keypoints.append(tracked_kps)
            
            # Create visualization
            if len(prev_frame.shape) == 2:
                prev_vis = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR)
                curr_vis = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2BGR)
            else:
                prev_vis = prev_frame.copy()
                curr_vis = curr_frame.copy()
            
            # Draw points and connections
            for j, idx in enumerate(tracked_indices):
                prev_kp = frame_keypoints[i-1][idx]
                curr_kp = tracked_kps[j]
                
                cv2.circle(prev_vis, (int(prev_kp.uv[0]), int(prev_kp.uv[1])), 
                          3, (0, 255, 0), -1)
                cv2.circle(curr_vis, (int(curr_kp.uv[0]), int(curr_kp.uv[1])), 
                          3, (0, 255, 0), -1)
            
            # Add frame info
            retention_ratio = len(tracked_kps) / len(frame_keypoints[i-1]) * 100
            cv2.putText(prev_vis, f"Frame {i-1}: {len(frame_keypoints[i-1])} keypoints", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(curr_vis, f"Frame {i}: {len(tracked_kps)} ({retention_ratio:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Create side-by-side display
            h, w = prev_vis.shape[:2]
            full_vis = np.zeros((h, w*2, 3), dtype=np.uint8)
            full_vis[:, :w] = prev_vis
            full_vis[:, w:] = curr_vis
            
            # Draw lines between matched points
            for j, idx in enumerate(tracked_indices):
                prev_kp = frame_keypoints[i-1][idx]
                curr_kp = tracked_kps[j]
                pt1 = (int(prev_kp.uv[0]), int(prev_kp.uv[1]))
                pt2 = (int(curr_kp.uv[0]) + w, int(curr_kp.uv[1]))
                cv2.line(full_vis, pt1, pt2, (0, 255, 255), 1)
            
            if SHOW_VISUALIZATIONS:
                cv2.imshow("Tracking Sequence", full_vis)
                key = cv2.waitKey(500)  # Wait for 500ms
                if key == 27:  # ESC
                    break
            
            if video_writer:
                video_writer.write(full_vis)
            
            prev_frame = curr_frame
        
        if video_writer:
            video_writer.release()
            
        if SHOW_VISUALIZATIONS:
            cv2.destroyAllWindows()
    
    def test_create_candidates_mask_with_visualization(self):
        """Test creation of mask for candidate keypoints with visualization."""
        feature_manager = FeatureManager()
        img = self.get_test_image(0)
        
        # Create some keypoints
        keypoints = [KeyPoint(np.array([100, 100], dtype=np.float32)), 
                     KeyPoint(np.array([200, 200], dtype=np.float32))]
        
        # Create mask
        mask = feature_manager.create_candidates_mask(img, keypoints, min_distance=20)
        
        # Visualize mask
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Convert to BGR if it's grayscale
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
                
            # Create a visualization showing original image, mask, and overlay
            h, w = img.shape[:2]
            vis = np.zeros((h, w*3, 3), dtype=np.uint8)
            
            # Original image with keypoints
            for kp in keypoints:
                cv2.circle(img_color, (int(kp.uv[0]), int(kp.uv[1])), 
                          20, (0, 0, 255), 2)  # Same size as min_distance
                cv2.circle(img_color, (int(kp.uv[0]), int(kp.uv[1])), 
                          5, (0, 255, 0), -1)  # Keypoint center
            
            vis[:, :w] = img_color
            
            # Mask (convert to color for visualization)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis[:, w:2*w] = mask_color
            
            # Overlay mask on image
            overlay = img_color.copy()
            overlay[mask == 0] = [0, 0, 255]  # Red areas are masked out
            vis[:, 2*w:] = overlay
            
            # Add labels
            cv2.putText(vis, "Original with Keypoints", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Mask (white=allowed)", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Overlay (red=masked)", (2*w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            show_or_save_image("Candidate Keypoints Mask", vis)
    
    def test_match_keypoints_with_visualization(self):
        """Test keypoint matching between frames with visualization."""
        feature_manager = FeatureManager(max_features=500)
        feature_manager.use_sift = True
        
        img1 = self.get_test_image(0)
        img2 = self.get_test_image(1)
        
        # Detect keypoints in both frames
        keypoints1 = feature_manager.detect_keypoints(img1)
        keypoints2 = feature_manager.detect_keypoints(img2)
        
        # Skip test if not enough keypoints with descriptors
        if not all([kp.des is not None for kp in keypoints1]) or not all([kp.des is not None for kp in keypoints2]):
            self.skipTest("Keypoints don't have descriptors for matching test")
        
        # Skip test if not enough keypoints
        if len(keypoints1) < 10 or len(keypoints2) < 10:
            self.skipTest("Not enough keypoints detected for matching test")
        
        # Match keypoints
        matches = feature_manager.match_keypoints(keypoints1, keypoints2)
        
        # Visualize matches
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Convert to BGR if it's grayscale
            if len(img1.shape) == 2:
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            else:
                img1_color = img1.copy()
                img2_color = img2.copy()
            
            # Create side-by-side display
            h, w = img1.shape[:2]
            vis = np.zeros((h, w*2, 3), dtype=np.uint8)
            vis[:, :w] = img1_color
            vis[:, w:] = img2_color
            
            # Draw all keypoints
            for kp in keypoints1:
                cv2.circle(vis, (int(kp.uv[0]), int(kp.uv[1])), 
                          3, (0, 0, 255), -1)
            
            for kp in keypoints2:
                cv2.circle(vis, (int(kp.uv[0]) + w, int(kp.uv[1])), 
                          3, (0, 0, 255), -1)
            
            # Draw matches
            for idx1, idx2 in matches:
                pt1 = (int(keypoints1[idx1].uv[0]), int(keypoints1[idx1].uv[1]))
                pt2 = (int(keypoints2[idx2].uv[0]) + w, int(keypoints2[idx2].uv[1]))
                
                # Draw larger circles for matched points
                cv2.circle(vis, pt1, 5, (0, 255, 0), -1)
                cv2.circle(vis, pt2, 5, (0, 255, 0), -1)
                
                # Draw connecting line
                cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            
            # Add text with match count
            cv2.putText(vis, f"Keypoints: {len(keypoints1)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis, f"Keypoints: {len(keypoints2)}", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis, f"Matches: {len(matches)}", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            show_or_save_image("SIFT Matches", vis)



if __name__ == '__main__':
    # Set visualization flags from command line args if provided
    if len(sys.argv) > 1:
        if "--no-viz" in sys.argv:
            SHOW_VISUALIZATIONS = False
            # Remove from args to not confuse unittest
            sys.argv.remove("--no-viz")
        
        if "--save-viz" in sys.argv:
            SAVE_VISUALIZATIONS = True
            # Remove from args to not confuse unittest
            sys.argv.remove("--save-viz")
            
        if "--viz-dir" in sys.argv:
            idx = sys.argv.index("--viz-dir")
            if idx + 1 < len(sys.argv):
                VIZ_OUTPUT_DIR = sys.argv[idx + 1]
                # Remove from args to not confuse unittest
                sys.argv.pop(idx)  # Remove --viz-dir
                sys.argv.pop(idx)  # Remove the directory path
    
    # Create output directory if saving
    if SAVE_VISUALIZATIONS:
        create_output_dir()
        
    unittest.main()