import unittest
import numpy as np
import cv2
import os
import sys
import time
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Add parent directory to path to import the modules
sys.path.append(str(Path(__file__).parent.parent))
from triangulator import Triangulator
from feature_manager import KeyPoint, Landmark
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

class TestTriangulator(unittest.TestCase):
    """Test cases for the Triangulator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with sample images and camera parameters."""
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
        
        # Create output directory for visualizations
        create_output_dir()
        
        # Create a synthetic camera calibration matrix if none from dataset
        if cls.dataset is None or not hasattr(cls.dataset, 'K'):
            print("Using synthetic camera calibration")
            cls.K = np.array([
                [718.8560, 0.0, 607.1928],
                [0.0, 718.8560, 185.2157],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            cls.K = cls.dataset.K
        
        # Create a default identity pose
        cls.pose1 = np.eye(4, dtype=np.float32)
        
        # Create a second pose with translation and rotation
        cls.pose2 = np.array([
            [0.9995, 0.0307, -0.0088, -0.5400],
            [-0.0307, 0.9995, 0.0028, 0.0044],
            [0.0089, -0.0025, 0.9999, 0.0019],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ], dtype=np.float32)
        
        # Create some synthetic 3D points and project them to both cameras
        cls.points_3d = np.array([
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [-1.0, 0.0, 5.0],
            [0.0, -1.0, 5.0],
            [2.0, 2.0, 7.0],
            [-2.0, 2.0, 7.0],
            [2.0, -2.0, 7.0],
            [-2.0, -2.0, 7.0]
        ], dtype=np.float32)
        
        # Project to camera 1
        cls.points_2d_1 = []
        for point in cls.points_3d:
            # Convert to homogeneous
            point_h = np.append(point, 1.0)
            # Project through camera 1
            P1 = cls.K @ cls.pose1[:3, :]
            point_proj = P1 @ point_h
            point_proj = point_proj[:2] / point_proj[2]
            cls.points_2d_1.append(KeyPoint(point_proj))
        
        # Project to camera 2
        cls.points_2d_2 = []
        for point in cls.points_3d:
            # Convert to homogeneous
            point_h = np.append(point, 1.0)
            # Project through camera 2
            P2 = cls.K @ cls.pose2[:3, :]
            point_proj = P2 @ point_h
            point_proj = point_proj[:2] / point_proj[2]
            cls.points_2d_2.append(KeyPoint(point_proj))
    
    def get_test_image(self, index=0):
        """Get a test image from the dataset."""
        if hasattr(self.__class__, 'dataset') and self.__class__.dataset is not None:
            img = self.__class__.dataset.get_image(index)
            if img is not None:
                return img
        
        # Create a synthetic image if dataset not available
        img = np.zeros((480, 640), dtype=np.uint8)
        for pt in self.__class__.points_2d_1:
            x, y = int(pt.uv[0]), int(pt.uv[1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 5, 255, -1)
        
        return img
    
    def test_linear_triangulation(self):
        """Test basic linear triangulation."""
        triangulator = Triangulator()
        
        # Triangulate using synthetic data
        landmarks = triangulator.linear_triangulation(
            self.__class__.K,
            self.__class__.pose1,
            self.__class__.pose2,
            self.__class__.points_2d_1,
            self.__class__.points_2d_2
        )
        
        # Check if landmarks were created
        self.assertEqual(len(landmarks), len(self.__class__.points_3d))
        
        # Verify triangulation accuracy
        for i, (landmark, true_point) in enumerate(zip(landmarks, self.__class__.points_3d)):
            # Allow for some error in triangulation
            error = np.linalg.norm(landmark.p - true_point)
            self.assertLess(error, 0.5, f"Triangulation error too large for point {i}: {error}")
        
        # Visualize the triangulation
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Create a 3D plot of true points and triangulated points
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot true points
            true_x = self.__class__.points_3d[:, 0]
            true_y = self.__class__.points_3d[:, 1]
            true_z = self.__class__.points_3d[:, 2]
            ax.scatter(true_x, true_y, true_z, color='blue', label='True Points')
            
            # Plot triangulated points
            triang_x = [lm.p[0] for lm in landmarks]
            triang_y = [lm.p[1] for lm in landmarks]
            triang_z = [lm.p[2] for lm in landmarks]
            ax.scatter(triang_x, triang_y, triang_z, color='red', label='Triangulated')
            
            # Plot camera positions
            cam1_pos = -self.__class__.pose1[:3, :3].T @ self.__class__.pose1[:3, 3]
            cam2_pos = -self.__class__.pose2[:3, :3].T @ self.__class__.pose2[:3, 3]
            
            ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], 
                      color='green', marker='^', s=100, label='Camera 1')
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], 
                      color='purple', marker='^', s=100, label='Camera 2')
            
            # Plot camera axes
            # Camera 1 axes
            axis_len = 0.5
            for i, color in enumerate(['r', 'g', 'b']):
                axis = np.zeros(3)
                axis[i] = axis_len
                axis_cam = self.__class__.pose1[:3, :3] @ axis
                ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
                         axis_cam[0], axis_cam[1], axis_cam[2], 
                         color=color)
            
            # Camera 2 axes
            for i, color in enumerate(['r', 'g', 'b']):
                axis = np.zeros(3)
                axis[i] = axis_len
                axis_cam = self.__class__.pose2[:3, :3] @ axis
                ax.quiver(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
                         axis_cam[0], axis_cam[1], axis_cam[2], 
                         color=color)
            
            # Connect true points to triangulated points
            for i in range(len(landmarks)):
                ax.plot([true_x[i], triang_x[i]], 
                        [true_y[i], triang_y[i]], 
                        [true_z[i], triang_z[i]], 
                        'k--', alpha=0.3)
            
            # Set axis labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title('Linear Triangulation Results')
            
            # Equal aspect ratio
            max_range = np.array([
                max(true_x) - min(true_x),
                max(true_y) - min(true_y),
                max(true_z) - min(true_z)
            ]).max() / 2.0
            
            mid_x = (max(true_x) + min(true_x)) * 0.5
            mid_y = (max(true_y) + min(true_y)) * 0.5
            mid_z = (max(true_z) + min(true_z)) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save or show plot
            if SAVE_VISUALIZATIONS:
                create_output_dir()
                plt_path = os.path.join(VIZ_OUTPUT_DIR, f"linear_triangulation_{int(time.time())}.png")
                plt.savefig(plt_path)
                print(f"Saved 3D plot to {plt_path}")
            
            if SHOW_VISUALIZATIONS:
                plt.show()
            else:
                plt.close()
    
    def test_check_triangulation(self):
        """Test triangulation verification with reprojection error checking."""
        triangulator = Triangulator()
        
        # Triangulate using synthetic data
        landmarks = triangulator.linear_triangulation(
            self.__class__.K,
            self.__class__.pose1,
            self.__class__.pose2,
            self.__class__.points_2d_1,
            self.__class__.points_2d_2
        )
        
        # Check triangulation results
        filtered_landmarks, filtered_pts1, filtered_pts2 = triangulator.check_triangulation(
            landmarks,
            self.__class__.K,
            self.__class__.pose1,
            self.__class__.pose2,
            self.__class__.points_2d_1,
            self.__class__.points_2d_2,
            max_error=1.0  # Strict error threshold
        )
        
        # Check if some landmarks passed the filter
        self.assertGreater(len(filtered_landmarks), 0)
        self.assertEqual(len(filtered_landmarks), len(filtered_pts1))
        self.assertEqual(len(filtered_landmarks), len(filtered_pts2))
        
        # Visualize the filtering results
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Create 2D plots showing reprojection
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Get image dimensions for plotting
            img = self.get_test_image(0)
            h, w = img.shape[:2]
            
            # Plot first camera view
            axs[0].set_xlim(0, w)
            axs[0].set_ylim(h, 0)  # Invert y-axis for image coordinates
            axs[0].set_title("Camera 1 View")
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")
            
            # Original points in blue
            original_pts1_x = [kp.uv[0] for kp in self.__class__.points_2d_1]
            original_pts1_y = [kp.uv[1] for kp in self.__class__.points_2d_1]
            axs[0].scatter(original_pts1_x, original_pts1_y, c='blue', marker='o', label='Original')
            
            # Reprojected points in red
            for lm in landmarks:
                pt3D_h = np.append(lm.p, 1.0)
                P1 = self.__class__.K @ self.__class__.pose1[:3, :]
                pt_proj = P1 @ pt3D_h
                pt_proj = pt_proj[:2] / pt_proj[2]
                axs[0].scatter(pt_proj[0], pt_proj[1], c='red', marker='x')
            
            # Filtered points in green
            filtered_pts1_x = [kp.uv[0] for kp in filtered_pts1]
            filtered_pts1_y = [kp.uv[1] for kp in filtered_pts1]
            axs[0].scatter(filtered_pts1_x, filtered_pts1_y, c='green', marker='+', label='Filtered')
            
            # Plot second camera view
            axs[1].set_xlim(0, w)
            axs[1].set_ylim(h, 0)  # Invert y-axis for image coordinates
            axs[1].set_title("Camera 2 View")
            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Y")
            
            # Original points in blue
            original_pts2_x = [kp.uv[0] for kp in self.__class__.points_2d_2]
            original_pts2_y = [kp.uv[1] for kp in self.__class__.points_2d_2]
            axs[1].scatter(original_pts2_x, original_pts2_y, c='blue', marker='o', label='Original')
            
            # Reprojected points in red
            for lm in landmarks:
                pt3D_h = np.append(lm.p, 1.0)
                P2 = self.__class__.K @ self.__class__.pose2[:3, :]
                pt_proj = P2 @ pt3D_h
                pt_proj = pt_proj[:2] / pt_proj[2]
                axs[1].scatter(pt_proj[0], pt_proj[1], c='red', marker='x')
            
            # Filtered points in green
            filtered_pts2_x = [kp.uv[0] for kp in filtered_pts2]
            filtered_pts2_y = [kp.uv[1] for kp in filtered_pts2]
            axs[1].scatter(filtered_pts2_x, filtered_pts2_y, c='green', marker='+', label='Filtered')
            
            # Add legend
            axs[0].legend()
            axs[1].legend()
            
            fig.suptitle("Triangulation Reprojection Check")
            plt.tight_layout()
            
            # Save or show plot
            if SAVE_VISUALIZATIONS:
                create_output_dir()
                plt_path = os.path.join(VIZ_OUTPUT_DIR, f"triangulation_check_{int(time.time())}.png")
                plt.savefig(plt_path)
                print(f"Saved reprojection plot to {plt_path}")
            
            if SHOW_VISUALIZATIONS:
                plt.show()
            else:
                plt.close()
    
    def test_nonlinear_triangulation(self):
        """Test non-linear triangulation refinement."""
        triangulator = Triangulator()
        
        # First perform linear triangulation
        landmarks = triangulator.linear_triangulation(
            self.__class__.K,
            self.__class__.pose1,
            self.__class__.pose2,
            self.__class__.points_2d_1,
            self.__class__.points_2d_2
        )
        
        # Then refine with non-linear optimization
        refined_landmarks = triangulator.nonlinear_triangulation(
            landmarks,
            self.__class__.K,
            self.__class__.pose1,
            self.__class__.pose2,
            self.__class__.points_2d_1,
            self.__class__.points_2d_2
        )
        
        # Check if refinement improved accuracy
        linear_errors = [np.linalg.norm(lm.p - true_pt) 
                         for lm, true_pt in zip(landmarks, self.__class__.points_3d)]
        nonlinear_errors = [np.linalg.norm(lm.p - true_pt) 
                           for lm, true_pt in zip(refined_landmarks, self.__class__.points_3d)]
        
        avg_linear_error = sum(linear_errors) / len(linear_errors)
        avg_nonlinear_error = sum(nonlinear_errors) / len(nonlinear_errors)
        
        print(f"Average linear triangulation error: {avg_linear_error:.4f}")
        print(f"Average nonlinear triangulation error: {avg_nonlinear_error:.4f}")
        
        # Nonlinear should generally be better or at least not much worse
        self.assertLessEqual(avg_nonlinear_error, avg_linear_error * 1.1)
        
        # Visualize the refinement
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot true points
            true_x = self.__class__.points_3d[:, 0]
            true_y = self.__class__.points_3d[:, 1]
            true_z = self.__class__.points_3d[:, 2]
            ax.scatter(true_x, true_y, true_z, color='blue', label='True Points')
            
            # Plot linear triangulation
            linear_x = [lm.p[0] for lm in landmarks]
            linear_y = [lm.p[1] for lm in landmarks]
            linear_z = [lm.p[2] for lm in landmarks]
            ax.scatter(linear_x, linear_y, linear_z, color='red', label='Linear')
            
            # Plot non-linear refinement
            nonlinear_x = [lm.p[0] for lm in refined_landmarks]
            nonlinear_y = [lm.p[1] for lm in refined_landmarks]
            nonlinear_z = [lm.p[2] for lm in refined_landmarks]
            ax.scatter(nonlinear_x, nonlinear_y, nonlinear_z, color='green', label='Non-linear')
            
            # Plot camera positions
            cam1_pos = -self.__class__.pose1[:3, :3].T @ self.__class__.pose1[:3, 3]
            cam2_pos = -self.__class__.pose2[:3, :3].T @ self.__class__.pose2[:3, 3]
            
            ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], 
                      color='black', marker='^', s=100, label='Camera 1')
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], 
                      color='gray', marker='^', s=100, label='Camera 2')
            
            # Connect points to show refinement vectors
            for i in range(len(landmarks)):
                ax.plot([linear_x[i], nonlinear_x[i]], 
                        [linear_y[i], nonlinear_y[i]], 
                        [linear_z[i], nonlinear_z[i]], 
                        'k-', alpha=0.5)
            
            # Set axis labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title('Non-linear Triangulation Refinement')
            
            # Add error information
            plt.figtext(0.02, 0.02, 
                       f"Linear error: {avg_linear_error:.4f}\nNon-linear error: {avg_nonlinear_error:.4f}", 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Equal aspect ratio
            max_range = np.array([
                max(true_x) - min(true_x),
                max(true_y) - min(true_y),
                max(true_z) - min(true_z)
            ]).max() / 2.0
            
            mid_x = (max(true_x) + min(true_x)) * 0.5
            mid_y = (max(true_y) + min(true_y)) * 0.5
            mid_z = (max(true_z) + min(true_z)) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save or show plot
            if SAVE_VISUALIZATIONS:
                create_output_dir()
                plt_path = os.path.join(VIZ_OUTPUT_DIR, f"nonlinear_triangulation_{int(time.time())}.png")
                plt.savefig(plt_path)
                print(f"Saved 3D plot to {plt_path}")
            
            if SHOW_VISUALIZATIONS:
                plt.show()
            else:
                plt.close()
    
    def test_triangulate_with_bearing_angle(self):
        """Test triangulation with bearing angle threshold."""
        triangulator = Triangulator()
        
        # Create a new pose with smaller baseline to test angle filtering
        smaller_baseline_pose = np.array([
            [0.9999, 0.0100, -0.0010, -0.1000],
            [-0.0100, 0.9999, 0.0000, 0.0010],
            [0.0010, 0.0000, 0.9999, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ], dtype=np.float32)
        
        # Triangulate with different bearing angle thresholds
        filtered_landmarks_5deg, filtered_pts1_5deg, filtered_pts2_5deg = triangulator.triangulate_with_bearing_angle(
            self.__class__.K,
            self.__class__.pose1,
            self.__class__.pose2,
            self.__class__.points_2d_1,
            self.__class__.points_2d_2,
            min_angle_deg=5.0
        )
        
        filtered_landmarks_1deg, filtered_pts1_1deg, filtered_pts2_1deg = triangulator.triangulate_with_bearing_angle(
            self.__class__.K,
            self.__class__.pose1,
            smaller_baseline_pose,
            self.__class__.points_2d_1,
            self.__class__.points_2d_2,
            min_angle_deg=1.0
        )
        
        # Check that angle threshold is working
        self.assertGreaterEqual(len(filtered_landmarks_1deg), len(filtered_landmarks_5deg))
        
        # Visualize bearing angles
        if SHOW_VISUALIZATIONS or SAVE_VISUALIZATIONS:
            # Calculate bearing angles for each point
            angles = []
            for pt_3d in self.__class__.points_3d:
                angle = triangulator.compute_bearing_angle(
                    pt_3d, self.__class__.pose1, self.__class__.pose2
                )
                angles.append(angle)
            
            angles_small = []
            for pt_3d in self.__class__.points_3d:
                angle = triangulator.compute_bearing_angle(
                    pt_3d, self.__class__.pose1, smaller_baseline_pose
                )
                angles_small.append(angle)
            
            # Create visualization
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot angles for normal baseline
            axs[0].bar(range(len(angles)), angles)
            axs[0].axhline(y=5.0, color='r', linestyle='--', label='5° threshold')
            axs[0].axhline(y=1.0, color='g', linestyle='--', label='1° threshold')
            axs[0].set_title("Normal Baseline")
            axs[0].set_xlabel("Point Index")
            axs[0].set_ylabel("Bearing Angle (degrees)")
            axs[0].set_ylim(0, max(angles) * 1.2)
            axs[0].legend()
            
            # Plot angles for small baseline
            axs[1].bar(range(len(angles_small)), angles_small)
            axs[1].axhline(y=5.0, color='r', linestyle='--', label='5° threshold')
            axs[1].axhline(y=1.0, color='g', linestyle='--', label='1° threshold')
            axs[1].set_title("Small Baseline")
            axs[1].set_xlabel("Point Index")
            axs[1].set_ylabel("Bearing Angle (degrees)")
            axs[1].set_ylim(0, max(angles) * 1.2)  # Use same scale
            axs[1].legend()
            
            fig.suptitle("Bearing Angles for Triangulation")
            plt.tight_layout()
            
            # Save or show plot
            if SAVE_VISUALIZATIONS:
                create_output_dir()
                plt_path = os.path.join(VIZ_OUTPUT_DIR, f"bearing_angles_{int(time.time())}.png")
                plt.savefig(plt_path)
                print(f"Saved bearing angle plot to {plt_path}")
            
            if SHOW_VISUALIZATIONS:
                plt.show()
            else:
                plt.close()
            
            # Create 3D visualization of the filtered points
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all 3D points
            true_x = self.__class__.points_3d[:, 0]
            true_y = self.__class__.points_3d[:, 1]
            true_z = self.__class__.points_3d[:, 2]
            ax.scatter(true_x, true_y, true_z, color='blue', alpha=0.5, label='All Points')
            
            # Plot points that passed the 5-degree filter with normal baseline
            if filtered_landmarks_5deg:
                filtered_x = [lm.p[0] for lm in filtered_landmarks_5deg]
                filtered_y = [lm.p[1] for lm in filtered_landmarks_5deg]
                filtered_z = [lm.p[2] for lm in filtered_landmarks_5deg]
                ax.scatter(filtered_x, filtered_y, filtered_z, color='red', 
                          label=f'Passed 5° ({len(filtered_landmarks_5deg)})')
            
            # Plot camera positions
            cam1_pos = -self.__class__.pose1[:3, :3].T @ self.__class__.pose1[:3, 3]
            cam2_pos = -self.__class__.pose2[:3, :3].T @ self.__class__.pose2[:3, 3]
            cam2_small_pos = -smaller_baseline_pose[:3, :3].T @ smaller_baseline_pose[:3, 3]
            
            ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], 
                      color='black', marker='^', s=100, label='Camera 1')
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], 
                      color='red', marker='^', s=100, label='Camera 2 (Normal)')
            ax.scatter([cam2_small_pos[0]], [cam2_small_pos[1]], [cam2_small_pos[2]], 
                      color='green', marker='^', s=100, label='Camera 2 (Small)')
            
            # Draw lines from Camera 1 to both Camera 2 positions to show baselines
            ax.plot([cam1_pos[0], cam2_pos[0]], 
                    [cam1_pos[1], cam2_pos[1]], 
                    [cam1_pos[2], cam2_pos[2]], 
                    'r--', label='Normal Baseline')
            
            ax.plot([cam1_pos[0], cam2_small_pos[0]], 
                    [cam1_pos[1], cam2_small_pos[1]], 
                    [cam1_pos[2], cam2_small_pos[2]], 
                    'g--', label='Small Baseline')
            
            # Set axis labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title('Triangulation with Bearing Angle Filter')
            
            # Adjust view for better visualization
            ax.view_init(elev=30, azim=45)
            
            # Save or show plot
            if SAVE_VISUALIZATIONS:
                create_output_dir()
                plt_path = os.path.join(VIZ_OUTPUT_DIR, f"bearing_angle_3d_{int(time.time())}.png")
                plt.savefig(plt_path)
                print(f"Saved 3D bearing angle plot to {plt_path}")
            
            if SHOW_VISUALIZATIONS:
                plt.show()
            else:
                plt.close()


if __name__ == '__main__':
    # Set visualization flags from command line args if provided

    if len(sys.argv) > 1:
        if "--no-viz" in sys.argv:
            SHOW_VISUALIZATIONS = False
            sys.argv.remove("--no-viz")
        
        if "--save-viz" in sys.argv:
            SAVE_VISUALIZATIONS = True
            sys.argv.remove("--save-viz")
            
        if "--viz-dir" in sys.argv:
            idx = sys.argv.index("--viz-dir")
            if idx + 1 < len(sys.argv):
                VIZ_OUTPUT_DIR = sys.argv[idx + 1]
                sys.argv.pop(idx)  # Remove --viz-dir
                sys.argv.pop(idx)  # Remove the directory path
    
    # Create output directory if saving
    if SAVE_VISUALIZATIONS:
        create_output_dir()
        
    unittest.main()