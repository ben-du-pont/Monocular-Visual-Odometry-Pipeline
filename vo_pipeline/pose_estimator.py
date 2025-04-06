import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class PoseEstimator:
    """
    A class for estimating camera poses from 2D-2D or 3D-2D correspondences.
    """
    def __init__(self):
        """
        Initialize the pose estimator.
        """
        pass
        
    def estimate_pose_2d2d(self, K, pts1, pts2, ransac_threshold=1.0, confidence=0.999):
        """
        Estimate relative camera pose from 2D-2D correspondences using the essential matrix.
        
        Args:
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            pts1 (list): List of KeyPoint objects in the first frame
            pts2 (list): List of KeyPoint objects in the second frame
            ransac_threshold (float, optional): RANSAC threshold in pixels
            confidence (float, optional): RANSAC confidence level
            
        Returns:
            tuple: (success, relative_pose, inliers)
                - success (bool): True if pose estimation was successful
                - relative_pose (numpy.ndarray): 4x4 transformation matrix
                - inliers (list): Indices of inlier correspondences
        """
        try:
            # Handle case with insufficient points
            if len(pts1) < 5:
                print("Insufficient points for essential matrix estimation (need at least 5)")
                return False, np.eye(4), []
                
            # Extract points
            pts1_array = np.array([kp.uv for kp in pts1], dtype=np.float32)
            pts2_array = np.array([kp.uv for kp in pts2], dtype=np.float32)
            
            # Debug information
            print(f"Initial shapes - pts1: {pts1_array.shape}, pts2: {pts2_array.shape}")
            
            # Check for invalid values
            if np.any(np.isnan(pts1_array)) or np.any(np.isinf(pts1_array)) or \
            np.any(np.isnan(pts2_array)) or np.any(np.isinf(pts2_array)):
                print("Point arrays contain NaN or Inf values")
                return False, np.eye(4), []
            
            # Ensure correct shape for OpenCV
            if len(pts1_array.shape) == 1:
                # Single point case - reshape to 1x2
                pts1_array = pts1_array.reshape(1, 2)
            elif pts1_array.shape[1] != 2:
                # Not Nx2 - try reshaping
                pts1_array = pts1_array.reshape(-1, 2)
                
            if len(pts2_array.shape) == 1:
                # Single point case - reshape to 1x2
                pts2_array = pts2_array.reshape(1, 2)
            elif pts2_array.shape[1] != 2:
                # Not Nx2 - try reshaping
                pts2_array = pts2_array.reshape(-1, 2)
            
            # Make sure arrays are contiguous (required by OpenCV)
            pts1_array = np.ascontiguousarray(pts1_array, dtype=np.float32)
            pts2_array = np.ascontiguousarray(pts2_array, dtype=np.float32)
            
            # Final check on shapes and types
            if pts1_array.shape != pts2_array.shape:
                print(f"Mismatched shapes after reshaping: pts1={pts1_array.shape}, pts2={pts2_array.shape}")
                return False, np.eye(4), []
                
            print(f"Final shapes - pts1: {pts1_array.shape}, pts2: {pts2_array.shape}, types: {pts1_array.dtype}, {pts2_array.dtype}")
            
            # Estimate essential matrix
            # Using RANSAC to robustly estimate despite outliers
            E, mask = cv2.findEssentialMat(
                pts1_array, pts2_array, K,
                method=cv2.RANSAC,
                prob=confidence,
                threshold=ransac_threshold
            )
        
            
            # Verify E is valid
            if E is None or mask is None or E.shape != (3, 3):
                print(f"Essential matrix estimation failed: {E.shape if E is not None else 'None'}")
                return False, np.eye(4), []
                
            # Get inlier indices
            inliers = np.where(mask.ravel() == 1)[0].tolist()
            
            # If we don't have enough inliers, return failure
            if len(inliers) < 5:
                print(f"Too few inliers found: {len(inliers)}")
                return False, np.eye(4), []
            
            # Filter points using the mask
            pts1_inliers = pts1_array[inliers]
            pts2_inliers = pts2_array[inliers]
            
            # Recover pose from essential matrix
            retval, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
            
            if not retval or R.shape != (3, 3) or t.shape[0] != 3:
                print("Pose recovery failed")
                return False, np.eye(4), []
            
            # Further filter points using the pose mask (points in front of camera)
            valid_indices = np.where(mask_pose.ravel() > 0)[0]
            if len(valid_indices) > 0:
                final_inliers = [inliers[i] for i in valid_indices]
            else:
                final_inliers = inliers  # Keep all if no valid triangulation
            
            # Construct 4x4 transformation matrix
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            
            print(f"Pose estimated with {len(final_inliers)}/{len(pts1)} inliers ({len(final_inliers)/len(pts1):.1%})")
            return True, T, final_inliers
            
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            import traceback
            traceback.print_exc()
            return False, np.eye(4), []
        
    def estimate_pose_3d2d(self, K, landmarks, keypoints, pnp_method=cv2.SOLVEPNP_ITERATIVE, 
                           ransac=True, reprojection_error=4.0, confidence=0.999, max_iter=100):
        """
        Estimate camera pose from 3D-2D correspondences using PnP.
        
        Args:
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            landmarks (list): List of Landmark objects
            keypoints (list): List of corresponding KeyPoint objects
            pnp_method (int, optional): PnP method to use
            ransac (bool, optional): Whether to use RANSAC
            reprojection_error (float, optional): Maximum reprojection error for RANSAC
            confidence (float, optional): RANSAC confidence level
            max_iter (int, optional): Maximum number of RANSAC iterations
            
        Returns:
            tuple: (success, pose, inliers)
                - success (bool): True if pose estimation was successful
                - pose (numpy.ndarray): 4x4 transformation matrix
                - inliers (list): Indices of inlier correspondences
        """
        try:
            # Check if we have enough correspondences
            if len(landmarks) < 4:
                print(f"Insufficient points for PnP (need at least 4, got {len(landmarks)})")
                return False, np.eye(4), []
                
            # Extract 3D points and 2D observations
            pts3D = np.array([l.p for l in landmarks], dtype=np.float32)
            pts2D = np.array([kp.uv for kp in keypoints], dtype=np.float32)
            
            # Reshape for OpenCV
            pts3D = pts3D.reshape(-1, 3)
            pts2D = pts2D.reshape(-1, 2)
            
            # Ensure data is contiguous for OpenCV
            pts3D = np.ascontiguousarray(pts3D)
            pts2D = np.ascontiguousarray(pts2D)
            
            # Estimate pose
            if ransac:
                # Use RANSAC for robust estimation
                # Handle potential API changes between OpenCV versions
                try:
                    # Start with a larger reprojection error for noisy data
                    ret_val = cv2.solvePnPRansac(
                        objectPoints=pts3D, 
                        imagePoints=pts2D, 
                        cameraMatrix=K, 
                        distCoeffs=None,
                        flags=pnp_method,
                        reprojectionError=reprojection_error * 2,  # Double the threshold for noisy data
                        iterationsCount=max_iter,
                        confidence=confidence
                    )
                    
                    # Unpack return values (different OpenCV versions return different tuples)
                    if len(ret_val) == 4:
                        success, rvec, tvec, inliers = ret_val
                    else:
                        success, rvec, tvec = ret_val[:3]
                        inliers = np.arange(len(landmarks)) if success else None
                        
                except Exception as e:
                    print(f"PnP RANSAC failed with error: {e}")
                    # Fallback to non-RANSAC method
                    success, rvec, tvec = cv2.solvePnP(
                        objectPoints=pts3D, 
                        imagePoints=pts2D, 
                        cameraMatrix=K, 
                        distCoeffs=None,
                        flags=pnp_method
                    )
                    inliers = np.arange(len(landmarks)) if success else None
                
            else:
                # Direct PnP without RANSAC
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=pts3D, 
                    imagePoints=pts2D, 
                    cameraMatrix=K, 
                    distCoeffs=None,
                    flags=pnp_method
                )
                
                inliers = np.arange(len(landmarks)) if success else None
                
            if not success or inliers is None:
                print("PnP estimation failed")
                return False, np.eye(4), []
                
            # Convert inliers to list
            if isinstance(inliers, np.ndarray):
                inliers = inliers.ravel().tolist()
            
            # If no inliers were found (edge case), use all points
            if len(inliers) == 0:
                inliers = list(range(len(landmarks)))
                
            # Refine pose using all inliers if we have enough
            if len(inliers) >= 4:
                pts3D_inliers = pts3D[inliers].reshape(-1, 3)
                pts2D_inliers = pts2D[inliers].reshape(-1, 2)
                
                try:
                    refine_success, refined_rvec, refined_tvec = cv2.solvePnP(
                        objectPoints=pts3D_inliers, 
                        imagePoints=pts2D_inliers, 
                        cameraMatrix=K, 
                        distCoeffs=None,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        useExtrinsicGuess=True,
                        rvec=rvec,
                        tvec=tvec
                    )
                    
                    if refine_success:
                        rvec = refined_rvec
                        tvec = refined_tvec
                        
                except Exception as e:
                    print(f"Refinement failed: {e}")
                
            # Convert to transformation matrix
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            
            inlier_percentage = len(inliers) / len(landmarks) * 100
            print(f"PnP estimated pose with {len(inliers)}/{len(landmarks)} inliers ({inlier_percentage:.1f}%)")
            return success, T, inliers
            
        except Exception as e:
            print(f"Error in PnP pose estimation: {e}")
            import traceback
            traceback.print_exc()
            return False, np.eye(4), []
            
    def compute_reprojection_errors(self, landmarks, keypoints, K, pose):
        """
        Compute reprojection errors for a set of 3D landmarks and their 2D observations.
        
        Args:
            landmarks (list): List of Landmark objects
            keypoints (list): List of corresponding KeyPoint objects
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            pose (numpy.ndarray): Camera pose (4x4 transformation matrix) - world to camera
            
        Returns:
            numpy.ndarray: Reprojection errors for each correspondence
        """
        if not landmarks or not keypoints:
            return np.array([])
            
        try:
            # Extract points
            pts3D = np.array([l.p for l in landmarks], dtype=np.float32)
            pts2D = np.array([kp.uv for kp in keypoints], dtype=np.float32)
            
            # Reshape for OpenCV
            pts3D = pts3D.reshape(-1, 3)
            pts2D = pts2D.reshape(-1, 2)
            
            # Extract rotation and translation
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Convert rotation to rotation vector
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(3, 1)
            
            # Use OpenCV's projectPoints function to reproject 3D points to 2D
            projected_pts, _ = cv2.projectPoints(pts3D, rvec, tvec, K, None)
            projected_pts = projected_pts.reshape(-1, 2)
            
            # Calculate reprojection errors
            errors = np.linalg.norm(pts2D - projected_pts, axis=1)
            
            return errors
            
        except Exception as e:
            print(f"Error in compute_reprojection_errors: {e}")
            import traceback
            traceback.print_exc()
            return np.array([float('inf')] * len(landmarks))
        

    def verify_pose(self, prev_pose, curr_pose, max_rotation_deg=30.0, max_translation=2.0):
        """
        Verify if the estimated pose is plausible by checking rotation and translation magnitudes.
        
        Args:
            prev_pose (numpy.ndarray): Previous camera pose (4x4)
            curr_pose (numpy.ndarray): Current camera pose (4x4)
            max_rotation_deg (float): Maximum allowed rotation in degrees
            max_translation (float): Maximum allowed translation
            
        Returns:
            bool: True if the pose is plausible, False otherwise
        """
        try:
            # Compute relative pose
            relative_pose = np.linalg.inv(prev_pose) @ curr_pose
            
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
            
            # Check if within bounds
            rotation_ok = max_angle_deg <= max_rotation_deg
            translation_ok = translation_mag <= max_translation
            
            return rotation_ok and translation_ok
            
        except Exception as e:
            print(f"Error in pose verification: {e}")
            return False
            
    def apply_motion_model(self, poses, n_poses=2):
        """
        Apply a simple motion model to predict the next pose.
        
        Args:
            poses (list): List of previous camera poses (4x4 matrices)
            n_poses (int): Number of poses to use for motion model
            
        Returns:
            numpy.ndarray: Predicted next pose (4x4)
        """
        try:
            # Need at least two poses for motion model
            if len(poses) < 2:
                return poses[-1].copy() if poses else np.eye(4)
                
            # Use the n most recent poses
            recent_poses = poses[-n_poses:]
            
            if len(recent_poses) < 2:
                return recent_poses[-1].copy()
                
            # Compute average motion
            avg_motion = np.eye(4)
            count = 0
            
            for i in range(1, len(recent_poses)):
                # Compute relative motion between consecutive poses
                relative_motion = np.linalg.inv(recent_poses[i-1]) @ recent_poses[i]
                
                if count == 0:
                    avg_motion = relative_motion.copy()
                else:
                    # Average the translations
                    avg_motion[:3, 3] += relative_motion[:3, 3]
                    
                    # For rotation, we can't just average the matrices
                    # But for small rotations, this approximation works okay
                    avg_motion[:3, :3] += relative_motion[:3, :3]
                
                count += 1
                
            if count > 1:
                # Normalize the average
                avg_motion[:3, 3] /= count
                
                # Normalize rotation (approximate)
                # For better results, use quaternions or exponential maps
                from scipy.linalg import polar
                R, S = polar(avg_motion[:3, :3])
                avg_motion[:3, :3] = R
            
            # Apply the average motion to the last pose
            predicted_pose = recent_poses[-1] @ avg_motion
            
            return predicted_pose
            
        except Exception as e:
            print(f"Error in motion model: {e}")
            return poses[-1].copy() if poses else np.eye(4)