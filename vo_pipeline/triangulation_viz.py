import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

class TriangulationVisualizer:
    """
    Visualizer for triangulation results.
    """
    
    def visualize_triangulation(self, landmarks, K, pose1, pose2, pts1, pts2, images=None):
        """
        Visualize triangulation results in 3D space and reprojection in 2D.
        
        Args:
            landmarks (list): List of Landmark objects
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            pose1 (numpy.ndarray): First camera pose (4x4)
            pose2 (numpy.ndarray): Second camera pose (4x4)
            pts1 (list): List of KeyPoint objects in the first frame
            pts2 (list): List of corresponding KeyPoint objects in the second frame
            images (tuple): Optional tuple of (image1, image2) for background display
        """
        # Create a figure with two subplots: 3D view and 2D reprojection
        fig = plt.figure(figsize=(20, 10))
        
        # 3D plot for camera poses and landmarks
        ax1 = fig.add_subplot(121, projection='3d')
        self._visualize_3d(ax1, landmarks, pose1, pose2)
        
        # 2D plot for original keypoints and reprojections
        ax2 = fig.add_subplot(122)
        self._visualize_reprojection(ax2, landmarks, K, pose1, pose2, pts1, pts2, images)
        
        plt.tight_layout()
        plt.savefig('triangulation_debug.png')
        plt.show()
    
    def _visualize_3d(self, ax, landmarks, pose1, pose2):
        """
        Visualize camera poses and landmarks in 3D space.
        """
        # Extract 3D points from landmarks
        points_3d = np.array([lm.p for lm in landmarks])
        
        # Plot 3D points
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='blue', marker='o', label='Triangulated Points')
        
        # Draw camera frustums for both cameras
        self._draw_camera(ax, pose1, 'Camera 1', color='r')
        self._draw_camera(ax, pose2, 'Camera 2', color='g')
        
        # Set equal aspect ratio for better 3D visualization
        max_range = np.array([
            points_3d[:, 0].max() - points_3d[:, 0].min(),
            points_3d[:, 1].max() - points_3d[:, 1].min(),
            points_3d[:, 2].max() - points_3d[:, 2].min()
        ]).max() / 2.0
        
        # Calculate mean position for centering
        mean_pos = np.mean(points_3d, axis=0)
        
        ax.set_xlim(mean_pos[0] - max_range, mean_pos[0] + max_range)
        ax.set_ylim(mean_pos[1] - max_range, mean_pos[1] + max_range)
        ax.set_zlim(mean_pos[2] - max_range, mean_pos[2] + max_range)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Triangulation')
        ax.legend()
        
        # Add camera position text
        cam1_center = -pose1[:3, :3].T @ pose1[:3, 3]
        cam2_center = -pose2[:3, :3].T @ pose2[:3, 3]
        ax.text(cam1_center[0], cam1_center[1], cam1_center[2], "Cam 1", color='r')
        ax.text(cam2_center[0], cam2_center[1], cam2_center[2], "Cam 2", color='g')
        
        # Add baseline length
        baseline = np.linalg.norm(cam2_center - cam1_center)
        ax.text(mean_pos[0], mean_pos[1], mean_pos[2] + max_range/2, 
               f"Baseline: {baseline:.2f}", bbox=dict(facecolor='white', alpha=0.7))
    
    def _draw_camera(self, ax, pose, label, color='r', scale=1.0):
        """
        Draw a camera frustum in 3D.
        """
        # Extract camera center and orientation
        R = pose[:3, :3]
        t = pose[:3, 3]
        center = -R.T @ t
        
        # Define camera frustum points (in camera coordinates)
        frustum_points = np.array([
            [0, 0, 0],  # Camera center
            [scale, scale, scale*2],  # Top-right
            [scale, -scale, scale*2],  # Bottom-right
            [-scale, -scale, scale*2],  # Bottom-left
            [-scale, scale, scale*2]   # Top-left
        ])
        
        # Transform to world coordinates
        R_world = R.T  # Transpose for camera-to-world
        frustum_world = np.array([R_world @ pt + center for pt in frustum_points])
        
        # Draw frustum edges
        for i in range(1, 5):
            ax.plot3D([frustum_world[0, 0], frustum_world[i, 0]], 
                     [frustum_world[0, 1], frustum_world[i, 1]], 
                     [frustum_world[0, 2], frustum_world[i, 2]], color)
        
        # Draw frustum face
        ax.plot3D([frustum_world[1, 0], frustum_world[2, 0], frustum_world[3, 0], 
                  frustum_world[4, 0], frustum_world[1, 0]], 
                 [frustum_world[1, 1], frustum_world[2, 1], frustum_world[3, 1], 
                  frustum_world[4, 1], frustum_world[1, 1]], 
                 [frustum_world[1, 2], frustum_world[2, 2], frustum_world[3, 2], 
                  frustum_world[4, 2], frustum_world[1, 2]], color)
        
        # Draw coordinate axes
        axis_length = scale * 2
        axes = np.array([
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, axis_length]   # Z-axis
        ])
        
        colors = ['r', 'g', 'b']  # X, Y, Z colors
        for i in range(3):
            ax.quiver(center[0], center[1], center[2],
                     R_world[0, i] * axis_length, 
                     R_world[1, i] * axis_length, 
                     R_world[2, i] * axis_length,
                     color=colors[i], arrow_length_ratio=0.1)
    
    def _visualize_reprojection(self, ax, landmarks, K, pose1, pose2, pts1, pts2, images=None):
        """
        Visualize original keypoints and reprojections in 2D.
        """
        # Create a side-by-side comparison of reprojections
        if images is None:
            # Create blank images if not provided
            img_height, img_width = 480, 640
            img1 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            img2 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        else:
            img1, img2 = images
            # Convert grayscale to color if needed
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2.shape) == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side display
        h, w = img1.shape[:2]
        combined_img = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined_img[:, :w] = img1
        combined_img[:, w:] = img2
        
        # Project 3D landmarks to both camera views
        points_3d = np.array([lm.p for lm in landmarks])
        
        # Projection matrices
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]
        
        # Homogeneous coordinates for 3D points
        points_3d_homo = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        
        # Project to both frames
        pts1_proj_homo = (P1 @ points_3d_homo.T).T
        pts2_proj_homo = (P2 @ points_3d_homo.T).T
        
        # Normalize by dividing by z
        pts1_proj = pts1_proj_homo[:, :2] / pts1_proj_homo[:, 2:3]
        pts2_proj = pts2_proj_homo[:, :2] / pts2_proj_homo[:, 2:3]
        
        # Extract original 2D points
        pts1_orig = np.array([kp.uv for kp in pts1])
        pts2_orig = np.array([kp.uv for kp in pts2])
        
        # Calculate reprojection errors
        errors1 = np.linalg.norm(pts1_proj - pts1_orig, axis=1)
        errors2 = np.linalg.norm(pts2_proj - pts2_orig, axis=1)
        
        # Draw original points, reprojection points, and error lines in the combined image
        for i in range(len(pts1_orig)):
            # Original points in both frames
            pt1_orig = tuple(map(int, pts1_orig[i]))
            pt2_orig = tuple(map(int, pts2_orig[i]))
            
            # Reprojection points
            pt1_proj = tuple(map(int, pts1_proj[i]))
            pt2_proj = tuple(map(int, pts2_proj[i] + np.array([w, 0])))  # Offset for image 2
            
            # Draw points
            cv2.circle(combined_img, pt1_orig, 3, (0, 255, 0), -1)  # Original in green
            cv2.circle(combined_img, (pt2_orig[0] + w, pt2_orig[1]), 3, (0, 255, 0), -1)
            
            cv2.circle(combined_img, pt1_proj, 3, (0, 0, 255), -1)  # Reprojection in red
            cv2.circle(combined_img, pt2_proj, 3, (0, 0, 255), -1)
            
            # Draw lines between original and reprojection - white for small errors, red for large
            error1_color = (0, 0, 255) if errors1[i] > 5.0 else (255, 255, 255)
            error2_color = (0, 0, 255) if errors2[i] > 5.0 else (255, 255, 255)
            
            cv2.line(combined_img, pt1_orig, pt1_proj, error1_color, 1)
            cv2.line(combined_img, (pt2_orig[0] + w, pt2_orig[1]), pt2_proj, error2_color, 1)
        
        # Display combined image
        ax.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        
        # Add error statistics
        mean_error1 = np.mean(errors1)
        mean_error2 = np.mean(errors2)
        max_error1 = np.max(errors1)
        max_error2 = np.max(errors2)
        
        error_text = (f"Camera 1 - Mean Error: {mean_error1:.2f}px, Max: {max_error1:.2f}px\n"
                     f"Camera 2 - Mean Error: {mean_error2:.2f}px, Max: {max_error2:.2f}px")
        
        ax.set_title(f"Reprojection Errors\n{error_text}")
    
    def visualize_reprojection_error_histogram(self, landmarks, K, pose1, pose2, pts1, pts2):
        """
        Visualize histogram of reprojection errors.
        """
        # Project 3D landmarks to both camera views
        points_3d = np.array([lm.p for lm in landmarks])
        
        # Projection matrices
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]
        
        # Homogeneous coordinates for 3D points
        points_3d_homo = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        
        # Project to both frames
        pts1_proj_homo = (P1 @ points_3d_homo.T).T
        pts2_proj_homo = (P2 @ points_3d_homo.T).T
        
        # Normalize by dividing by z
        pts1_proj = pts1_proj_homo[:, :2] / pts1_proj_homo[:, 2:3]
        pts2_proj = pts2_proj_homo[:, :2] / pts2_proj_homo[:, 2:3]
        
        # Extract original 2D points
        pts1_orig = np.array([kp.uv for kp in pts1])
        pts2_orig = np.array([kp.uv for kp in pts2])
        
        # Calculate reprojection errors
        errors1 = np.linalg.norm(pts1_proj - pts1_orig, axis=1)
        errors2 = np.linalg.norm(pts2_proj - pts2_orig, axis=1)
        
        # Create a figure with two histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram for camera 1
        bins1 = [0, 1, 2, 5, 10, 20, 50, 100, 1000]
        ax1.hist(errors1, bins=bins1, alpha=0.7, log=True)
        ax1.set_title(f"Camera 1 Reprojection Errors\nMean: {np.mean(errors1):.2f}px")
        ax1.set_xlabel("Reprojection error (pixels)")
        ax1.set_ylabel("Count (log scale)")
        ax1.grid(True, alpha=0.3)
        
        # Histogram for camera 2
        ax2.hist(errors2, bins=bins1, alpha=0.7, log=True)
        ax2.set_title(f"Camera 2 Reprojection Errors\nMean: {np.mean(errors2):.2f}px")
        ax2.set_xlabel("Reprojection error (pixels)")
        ax2.set_ylabel("Count (log scale)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reprojection_error_histogram.png')
        plt.show()
        
        # Print detailed statistics
        print("\nReprojection Error Statistics:")
        print(f"Camera 1 - Mean: {np.mean(errors1):.2f}px, Median: {np.median(errors1):.2f}px, "
              f"Min: {np.min(errors1):.2f}px, Max: {np.max(errors1):.2f}px")
        print(f"Camera 2 - Mean: {np.mean(errors2):.2f}px, Median: {np.median(errors2):.2f}px, "
              f"Min: {np.min(errors2):.2f}px, Max: {np.max(errors2):.2f}px")
        
        # Print distribution by range
        ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, float('inf'))]
        print("\nCamera 1 Error Distribution:")
        for r_min, r_max in ranges:
            count = np.sum((errors1 >= r_min) & (errors1 < r_max))
            pct = 100 * count / len(errors1)
            print(f"  {r_min}-{r_max if r_max != float('inf') else '∞'}px: {count} points ({pct:.1f}%)")
        
        print("\nCamera 2 Error Distribution:")
        for r_min, r_max in ranges:
            count = np.sum((errors2 >= r_min) & (errors2 < r_max))
            pct = 100 * count / len(errors2)
            print(f"  {r_min}-{r_max if r_max != float('inf') else '∞'}px: {count} points ({pct:.1f}%)")
    
    def visualize_coordinate_systems(self, K, pose1, pose2):
        """
        Visualize the camera coordinate systems in 3D.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw coordinate systems for both cameras
        self._draw_camera(ax, pose1, 'Camera 1', color='r', scale=0.5)
        self._draw_camera(ax, pose2, 'Camera 2', color='g', scale=0.5)
        
        # Draw world origin
        ax.scatter([0], [0], [0], c='k', marker='o', s=100, label='World Origin')
        
        # Draw axes for world coordinate system
        axis_length = 1.0
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1)
        
        # Calculate camera centers
        C1 = -pose1[:3, :3].T @ pose1[:3, 3]
        C2 = -pose2[:3, :3].T @ pose2[:3, 3]
        
        # Add camera position text
        ax.text(C1[0], C1[1], C1[2], "Cam 1", color='r')
        ax.text(C2[0], C2[1], C2[2], "Cam 2", color='g')
        
        # Add baseline information
        baseline = np.linalg.norm(C2 - C1)
        ax.text(0, 0, 1.5, f"Baseline: {baseline:.2f}", 
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Add pose matrices information in text box
        pose1_txt = f"Pose1:\n{pose1[0,0]:.2f} {pose1[0,1]:.2f} {pose1[0,2]:.2f} {pose1[0,3]:.2f}\n" \
                   f"{pose1[1,0]:.2f} {pose1[1,1]:.2f} {pose1[1,2]:.2f} {pose1[1,3]:.2f}\n" \
                   f"{pose1[2,0]:.2f} {pose1[2,1]:.2f} {pose1[2,2]:.2f} {pose1[2,3]:.2f}"
        
        pose2_txt = f"Pose2:\n{pose2[0,0]:.2f} {pose2[0,1]:.2f} {pose2[0,2]:.2f} {pose2[0,3]:.2f}\n" \
                   f"{pose2[1,0]:.2f} {pose2[1,1]:.2f} {pose2[1,2]:.2f} {pose2[1,3]:.2f}\n" \
                   f"{pose2[2,0]:.2f} {pose2[2,1]:.2f} {pose2[2,2]:.2f} {pose2[2,3]:.2f}"
        
        ax.text2D(0.05, 0.15, pose1_txt, transform=ax.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7))
        ax.text2D(0.05, 0.05, pose2_txt, transform=ax.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Set equal aspect ratio
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Coordinate Systems')
        ax.legend()
        
        plt.savefig('coordinate_systems.png')
        plt.show()