import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from feature_manager import Landmark


class Triangulator:
    """
    A class for triangulating 3D points from 2D correspondences.
    """
    def __init__(self):
        """
        Initialize the triangulator.
        """
        pass
        
    def linear_triangulation(self, K, pose1, pose2, pts1, pts2):
        """
        Triangulate points using linear method (DLT) with camera-to-world poses.
        
        Args:
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            pose1 (numpy.ndarray): First camera pose (camera-to-world, 4x4)
            pose2 (numpy.ndarray): Second camera pose (camera-to-world, 4x4)
            pts1 (list): List of KeyPoint objects in the first frame
            pts2 (list): List of corresponding KeyPoint objects in the second frame
            
        Returns:
            list: List of Landmark objects
        """
        # Validate inputs
        assert K.shape == (3, 3), f"Camera matrix K must be 3x3, got {K.shape}"
        assert pose1.shape == (4, 4), f"Pose1 must be 4x4, got {pose1.shape}"
        assert pose2.shape == (4, 4), f"Pose2 must be 4x4, got {pose2.shape}"
        assert len(pts1) == len(pts2), f"Point lists must have same length, got {len(pts1)} and {len(pts2)}"
        
        if len(pts1) == 0:
            return []
        
        # Extract points
        pts1_array = np.array([kp.uv for kp in pts1], dtype=np.float32)
        pts2_array = np.array([kp.uv for kp in pts2], dtype=np.float32)
        
        # Ensure correct shape
        if pts1_array.shape[1] != 2 or pts2_array.shape[1] != 2:
            pts1_array = pts1_array.reshape(-1, 2)
            pts2_array = pts2_array.reshape(-1, 2)
        
        # Convert camera-to-world poses to world-to-camera for projection matrices
        pose1_wc = np.linalg.inv(pose1)
        pose2_wc = np.linalg.inv(pose2)
        
        # Construct projection matrices
        P1 = K @ pose1_wc[:3, :]
        P2 = K @ pose2_wc[:3, :]
        
        # Triangulate points
        pts4D = cv2.triangulatePoints(P1, P2, pts1_array.T, pts2_array.T)
        pts3D = (pts4D[:3] / pts4D[3]).T
        
        # Create landmarks
        landmarks = [
            Landmark(
                pts3D[i], 
                pts1[i].des if hasattr(pts1[i], 'des') and pts1[i].des is not None else None,
                pts1[i].t_first
            ) 
            for i in range(len(pts3D))
        ]
        
        return landmarks
        
    def check_triangulation(self, landmarks, K, pose1, pose2, pts1, pts2, max_error=2.0):
        """
        Check triangulation results based on reprojection error - vectorized version.
        
        Args:
            landmarks (list): List of Landmark objects
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            pose1 (numpy.ndarray): First camera pose (4x4)
            pose2 (numpy.ndarray): Second camera pose (4x4)
            pts1 (list): List of KeyPoint objects in the first frame
            pts2 (list): List of corresponding KeyPoint objects in the second frame
            max_error (float, optional): Maximum allowed reprojection error
            
        Returns:
            tuple: (landmarks, pts1, pts2) after filtering
            
        Raises:
            AssertionError: If inputs have invalid shapes or sizes
        """
        if not landmarks:
            return [], [], []
            
        # Validate inputs
        assert K.shape == (3, 3), f"Camera matrix K must be 3x3, got {K.shape}"
        assert pose1.shape == (4, 4), f"Pose1 must be 4x4, got {pose1.shape}"
        assert pose2.shape == (4, 4), f"Pose2 must be 4x4, got {pose2.shape}"
        assert len(landmarks) == len(pts1) == len(pts2), "All lists must have the same length"
        
        # Convert camera-to-world poses to world-to-camera for projection
        pose1_wc = np.linalg.inv(pose1)
        pose2_wc = np.linalg.inv(pose2)
        
        # Compute projection matrices correctly
        P1 = K @ pose1_wc[:3, :]
        P2 = K @ pose2_wc[:3, :]
        
        # Extract camera centers correctly for camera-to-world poses
        C1 = pose1[:3, 3]  # Camera center is directly in the translation part
        C2 = pose2[:3, 3]
            
        # Extract all 3D points as a single array
        points_3D = np.array([landmark.p for landmark in landmarks])
        
        # Create homogeneous coordinates
        points_3D_homo = np.hstack((points_3D, np.ones((len(points_3D), 1))))
        
        # Project all points to both frames at once
        pts1_proj_homo = (P1 @ points_3D_homo.T).T
        pts2_proj_homo = (P2 @ points_3D_homo.T).T
        
        # Normalize by dividing by z-coordinate
        pts1_proj = pts1_proj_homo[:, :2] / pts1_proj_homo[:, 2:3]
        pts2_proj = pts2_proj_homo[:, :2] / pts2_proj_homo[:, 2:3]
        
        # Extract the original 2D points
        pts1_orig = np.array([kp.uv for kp in pts1])
        pts2_orig = np.array([kp.uv for kp in pts2])
        
        # Compute reprojection errors
        errors1 = np.linalg.norm(pts1_proj - pts1_orig, axis=1)
        errors2 = np.linalg.norm(pts2_proj - pts2_orig, axis=1)
        avg_errors = (errors1 + errors2) / 2
        
        # Check if points are in front of both cameras
        # Compute vectors from camera center to 3D points
        vectors_to_cam1 = points_3D - C1
        vectors_to_cam2 = points_3D - C2
        
        # Transform points to camera coordinates correctly
        points_in_cam1 = (pose1_wc[:3, :3] @ points_3D.T).T + pose1_wc[:3, 3]
        points_in_cam2 = (pose2_wc[:3, :3] @ points_3D.T).T + pose2_wc[:3, 3]

        
        # Check z-coordinate (depth)
        in_front_of_cameras = (points_in_cam1[:, 2] > 0) & (points_in_cam2[:, 2] > 0)
        
        # Create filter mask
        valid_mask = (avg_errors < max_error) & in_front_of_cameras
        
        # Store reprojection errors in landmarks
        for i, landmark in enumerate(landmarks):
            landmark.reprojection_error = avg_errors[i]
        
        # Filter based on mask
        filtered_landmarks = [landmarks[i] for i in range(len(landmarks)) if valid_mask[i]]
        filtered_pts1 = [pts1[i] for i in range(len(pts1)) if valid_mask[i]]
        filtered_pts2 = [pts2[i] for i in range(len(pts2)) if valid_mask[i]]
        
        print(f"Initial landmarks: {len(landmarks)}")
        print(f"Valid after reprojection+depth: {np.sum(valid_mask)}")
        print(f"  Avg reprojection error mean: {avg_errors.mean():.2f}")
        print(f"  Avg reprojection error max: {avg_errors.max():.2f}")
        print(f"  Avg reprojection error min: {avg_errors.min():.2f}")
        print(f" Median reprojection error: {np.median(avg_errors):.2f}")
        print(f"  In front of both cameras: {np.sum((points_in_cam1[:,2] > 0) & (points_in_cam2[:,2] > 0))}")

        # # Create visualizer instance
        # visualizer = TriangulationVisualizer()

        # # Visualize camera coordinate systems
        # visualizer.visualize_coordinate_systems(K, pose1, pose2)

        # # Visualize 3D triangulation and reprojection
        # visualizer.visualize_triangulation(landmarks, K, pose1, pose2, pts1, pts2)#, images=(frame1, frame2) if 'frame1' in locals() and 'frame2' in locals() else None)

        # # Visualize reprojection error histogram
        # visualizer.visualize_reprojection_error_histogram(landmarks, K, pose1, pose2, pts1, pts2)

        return filtered_landmarks, filtered_pts1, filtered_pts2
        
    def nonlinear_triangulation(self, landmarks, K, pose1, pose2, pts1, pts2, max_iter=10):
        """
        Refine triangulation using non-linear optimization with batch processing.
        
        Args:
            landmarks (list): List of Landmark objects to refine
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            pose1 (numpy.ndarray): First camera pose (4x4)
            pose2 (numpy.ndarray): Second camera pose (4x4)
            pts1 (list): List of KeyPoint objects in the first frame
            pts2 (list): List of corresponding KeyPoint objects in the second frame
            max_iter (int, optional): Maximum number of iterations
            
        Returns:
            list: Refined Landmark objects
            
        Raises:
            AssertionError: If inputs have invalid shapes or sizes
        """
        if not landmarks:
            return []
            
        # Validate inputs
        assert K.shape == (3, 3), f"Camera matrix K must be 3x3, got {K.shape}"
        assert pose1.shape == (4, 4), f"Pose1 must be 4x4, got {pose1.shape}"
        assert pose2.shape == (4, 4), f"Pose2 must be 4x4, got {pose2.shape}"
        assert len(landmarks) == len(pts1) == len(pts2), "All lists must have the same length"
        
        # Convert camera-to-world poses to world-to-camera for projection
        pose1_wc = np.linalg.inv(pose1)
        pose2_wc = np.linalg.inv(pose2)
        
        # Precompute projection matrices correctly
        P1 = K @ pose1_wc[:3, :]
        P2 = K @ pose2_wc[:3, :]
            
        # Define the reprojection error function
        def reprojection_error(x, K, P1, P2, pt1, pt2):
            # Extract 3D point
            pt3D = x.reshape(3)
            
            # Project to frame 1
            pt3D_homo = np.append(pt3D, 1)
            pt1_proj_homo = P1 @ pt3D_homo
            pt1_proj = pt1_proj_homo[:2] / pt1_proj_homo[2]
            
            # Project to frame 2
            pt2_proj_homo = P2 @ pt3D_homo
            pt2_proj = pt2_proj_homo[:2] / pt2_proj_homo[2]
            
            # Compute reprojection errors
            error1 = pt1_proj - pt1.uv
            error2 = pt2_proj - pt2.uv
            
            return np.concatenate([error1, error2])
            
        # Refine each landmark - can't easily vectorize due to least_squares API
        refined_landmarks = []
        
        for i, landmark in enumerate(landmarks):
            # Initial 3D point
            x0 = landmark.p.flatten()
            
            # Optimize
            result = least_squares(
                reprojection_error, x0, 
                args=(K, P1, P2, pts1[i], pts2[i]),
                method='lm',
                max_nfev=max_iter
            )
            
            # Update landmark with refined position
            refined_landmark = Landmark(
                result.x, landmark.des, landmark.t_first
            )
            refined_landmark.reprojection_error = np.linalg.norm(result.fun) / 2
            refined_landmarks.append(refined_landmark)
            
        return refined_landmarks
        
    def triangulate_with_bearing_angle(self, K, pose1, pose2, pts1, pts2, min_angle_deg=3.0, max_error=2.0):
        """
        Triangulate points with bearing angle check to ensure good triangulation.
        
        Args:
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            pose1 (numpy.ndarray): First camera pose (4x4)
            pose2 (numpy.ndarray): Second camera pose (4x4)
            pts1 (list): List of KeyPoint objects in the first frame
            pts2 (list): List of corresponding KeyPoint objects in the second frame
            min_angle_deg (float, optional): Minimum bearing angle in degrees
            max_error (float, optional): Maximum allowed reprojection error
            
        Returns:
            tuple: (landmarks, pts1, pts2) after filtering
            
        Raises:
            AssertionError: If inputs have invalid shapes or sizes
        """
        # Validate inputs
        assert K.shape == (3, 3), f"Camera matrix K must be 3x3, got {K.shape}"
        assert pose1.shape == (4, 4), f"Pose1 must be 4x4, got {pose1.shape}"
        assert pose2.shape == (4, 4), f"Pose2 must be 4x4, got {pose2.shape}"
        assert len(pts1) == len(pts2), f"Point lists must have same length, got {len(pts1)} and {len(pts2)}"
        
        if len(pts1) == 0:
            return [], [], []
        
        # Triangulate points
        landmarks = self.linear_triangulation(K, pose1, pose2, pts1, pts2)
        
        if not landmarks:
            print("No landmarks triangulated in linear triangulation")
            return [], [], []
        print(f"Total triangulated: {len(landmarks)}")

        # Check triangulation based on reprojection error
        landmarks, pts1, pts2 = self.check_triangulation(
            landmarks, K, pose1, pose2, pts1, pts2, max_error
        )
        
        if not landmarks:
            print("No landmarks passed reprojection error check")
            return [], [], []
        print(f"Total triangulated: {len(landmarks)}")
        # Refine using non-linear optimization
        landmarks = self.nonlinear_triangulation(
            landmarks, K, pose1, pose2, pts1, pts2
        )
        
        # Extract camera centers once
        C1 = -pose1[:3, :3].T @ pose1[:3, 3]
        C2 = -pose2[:3, :3].T @ pose2[:3, 3]
        
        # Vectorized bearing angle calculation
        points_3D = np.array([landmark.p for landmark in landmarks])
        
        # Vectors from camera centers to 3D points
        v1 = points_3D - C1
        v2 = points_3D - C2
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Compute angle between vectors
        # Using dot product: cos(angle) = v1·v2 / (|v1|·|v2|)
        dot_products = np.sum(v1 * v2, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)  # Ensure valid range for arccos
        angles_deg = np.degrees(np.arccos(dot_products))
        
        # Transform points to camera coordinates correctly
        pose1_wc = np.linalg.inv(pose1)
        pose2_wc = np.linalg.inv(pose2)
        points_in_cam1 = (pose1_wc[:3, :3] @ points_3D.T).T + pose1_wc[:3, 3]
        points_in_cam2 = (pose2_wc[:3, :3] @ points_3D.T).T + pose2_wc[:3, 3]
        
        in_front_of_cameras = (points_in_cam1[:, 2] > 0) & (points_in_cam2[:, 2] > 0)
        
        # Filter based on bearing angle and depth
        valid_mask = (angles_deg >= min_angle_deg) & in_front_of_cameras
        
        filtered_landmarks = [landmarks[i] for i in range(len(landmarks)) if valid_mask[i]]
        filtered_pts1 = [pts1[i] for i in range(len(pts1)) if valid_mask[i]]
        filtered_pts2 = [pts2[i] for i in range(len(pts2)) if valid_mask[i]]
        
        print(f"Total triangulated: {len(landmarks)}")
        print(f"Passed reprojection: {len(pts1)}")
        print(f"Points in front of both cameras: {np.sum(in_front_of_cameras)}")
        print(f"Passed bearing angle: {np.sum(angles_deg >= min_angle_deg)}")
        print(f"Final valid landmarks: {np.sum(valid_mask)}")


        print(f"Filtered landmarks based on bearing angle: {len(filtered_landmarks)}/{len(landmarks)}")
        return filtered_landmarks, filtered_pts1, filtered_pts2
            
    def compute_bearing_angle(self, point_3d, camera_pose1, camera_pose2):
        """
        Compute the bearing angle between two camera views of a 3D point.
        
        Args:
            point_3d (numpy.ndarray): 3D point coordinates
            camera_pose1 (numpy.ndarray): First camera pose (4x4)
            camera_pose2 (numpy.ndarray): Second camera pose (4x4)
            
        Returns:
            float: Bearing angle in degrees
            
        Raises:
            AssertionError: If inputs have invalid shapes
        """
        # Validate inputs
        assert camera_pose1.shape == (4, 4), f"Camera pose 1 must be 4x4, got {camera_pose1.shape}"
        assert camera_pose2.shape == (4, 4), f"Camera pose 2 must be 4x4, got {camera_pose2.shape}"
        
        # Extract camera centers
        C1 = camera_pose1[:3, 3]
        C2 = camera_pose2[:3, 3]
        
        # Vectors from camera centers to 3D point
        v1 = point_3d - C1
        v2 = point_3d - C2
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute angle between vectors
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        return angle_deg