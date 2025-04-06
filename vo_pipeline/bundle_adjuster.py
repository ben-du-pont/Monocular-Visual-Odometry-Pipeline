import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
from scipy.spatial.transform import Rotation

class BundleAdjuster:
    """
    A class for bundle adjustment optimization to refine camera poses and 3D landmarks.
    """
    def __init__(self, method='trf', ftol=1e-4, xtol=1e-4, loss='cauchy'):
        """
        Initialize the bundle adjuster.
        
        Args:
            method (str, optional): Optimization method ('trf', 'dogleg', etc.)
            ftol (float, optional): Function tolerance for termination
            xtol (float, optional): Parameter tolerance for termination
            loss (str, optional): Loss function ('linear', 'huber', 'cauchy', etc.)
        """
        self.method = method
        self.ftol = ftol
        self.xtol = xtol
        self.loss = loss
        
    def local_bundle_adjustment(self, poses, landmarks, observations, K, window_size=10, fixed_first=True):
        """
        Perform local bundle adjustment on a window of recent frames.
        
        Args:
            poses (list): List of all camera poses (4x4 transformation matrices)
            landmarks (dict): Dictionary mapping landmark IDs to Landmark objects
            observations (dict): Dictionary mapping (frame_idx, landmark_id) to 2D observations
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            window_size (int, optional): Size of the window for local BA
            fixed_first (bool, optional): Whether to fix the first pose in the window
            
        Returns:
            tuple: (updated_poses, updated_landmarks)
        """
        if len(poses) < 2:
            return poses, landmarks
            
        # Select the window of poses to optimize
        start_idx = max(0, len(poses) - window_size)
        window_poses = poses[start_idx:]
        num_poses = len(window_poses)
        
        # Get landmark observations within the window
        window_observations = {}
        window_landmarks = {}
        
        for (frame_idx, landmark_id), observation in observations.items():
            if start_idx <= frame_idx < start_idx + num_poses and landmark_id in landmarks:
                window_observations[(frame_idx - start_idx, landmark_id)] = observation
                window_landmarks[landmark_id] = landmarks[landmark_id]
                
        # Return early if there are not enough observations
        if len(window_observations) < 10 or len(window_landmarks) < 5:
            return poses, landmarks
            
        # Build the parameter vector
        param_vec, param_indices = self._build_parameters(
            window_poses, window_landmarks, fixed_first
        )
        
        # Define the optimization function
        def bundle_adjustment_objective(params):
            # Reconstruct poses and landmarks from the parameter vector
            current_poses, current_landmarks = self._reconstruct_parameters(
                params, param_indices, window_poses, window_landmarks, fixed_first
            )
            
            # Compute reprojection errors
            errors = []
            
            for (frame_idx, landmark_id), observation in window_observations.items():
                if landmark_id in current_landmarks:
                    # Get the 3D point
                    point3D = current_landmarks[landmark_id].p.reshape(3)
                    
                    # Get the pose
                    pose = current_poses[frame_idx]
                    
                    # Project the 3D point to the image
                    projected = self._project_point(point3D, pose, K)
                    
                    # Compute reprojection error
                    error = projected - observation.reshape(2)
                    errors.append(error)
                    
            return np.array(errors).flatten()
            
        # Define the Jacobian sparsity structure
        jac_sparsity = self._build_jacobian_sparsity(
            param_indices, window_observations, window_landmarks, fixed_first
        )
        
        # Run the optimization
        result = opt.least_squares(
            bundle_adjustment_objective,
            param_vec,
            jac_sparsity=jac_sparsity,
            method=self.method,
            ftol=self.ftol,
            xtol=self.xtol,
            loss=self.loss
        )
        
        # Reconstruct the optimized poses and landmarks
        optimized_poses, optimized_landmarks = self._reconstruct_parameters(
            result.x, param_indices, window_poses, window_landmarks, fixed_first
        )
        
        # Update the global poses and landmarks
        updated_poses = poses.copy()
        for i in range(num_poses):
            if i > 0 or not fixed_first:  # Skip the first pose if it's fixed
                updated_poses[start_idx + i] = optimized_poses[i]
                
        updated_landmarks = landmarks.copy()
        for landmark_id, landmark in optimized_landmarks.items():
            updated_landmarks[landmark_id] = landmark
            
        return updated_poses, updated_landmarks
        
    def _build_parameters(self, poses, landmarks, fixed_first=True):
        """
        Build the parameter vector for optimization.
        
        Args:
            poses (list): List of camera poses
            landmarks (dict): Dictionary of landmarks
            fixed_first (bool): Whether to fix the first pose
            
        Returns:
            tuple: (parameter vector, parameter indices)
        """
        param_vec = []
        param_indices = {
            'poses': {},
            'landmarks': {}
        }
        
        # Add poses to the parameter vector
        start_idx = 1 if fixed_first else 0
        for i in range(start_idx, len(poses)):
            pose = poses[i]
            
            # Convert rotation matrix to quaternion (4 parameters)
            R = pose[:3, :3]
            r = Rotation.from_matrix(R)
            quat = r.as_quat()  # x, y, z, w
            
            # Extract translation (3 parameters)
            t = pose[:3, 3]
            
            # Add to parameter vector
            param_idx = len(param_vec)
            param_vec.extend(quat)
            param_vec.extend(t)
            
            # Store index
            param_indices['poses'][i] = (param_idx, param_idx + 7)  # 4 for quat, 3 for translation
            
        # Add landmarks to the parameter vector
        for landmark_id, landmark in landmarks.items():
            param_idx = len(param_vec)
            param_vec.extend(landmark.p.reshape(3))
            
            # Store index
            param_indices['landmarks'][landmark_id] = (param_idx, param_idx + 3)
            
        return np.array(param_vec), param_indices
        
    def _reconstruct_parameters(self, params, param_indices, original_poses, original_landmarks, fixed_first=True):
        """
        Reconstruct poses and landmarks from the parameter vector.
        
        Args:
            params (numpy.ndarray): Parameter vector
            param_indices (dict): Parameter indices
            original_poses (list): Original camera poses
            original_landmarks (dict): Original landmarks
            fixed_first (bool): Whether the first pose is fixed
            
        Returns:
            tuple: (poses, landmarks)
        """
        # Reconstruct poses
        poses = original_poses.copy()
        for i, pose in enumerate(poses):
            if i > 0 or not fixed_first:  # Skip the first pose if it's fixed
                if i in param_indices['poses']:
                    start_idx, end_idx = param_indices['poses'][i]
                    
                    # Extract quaternion and translation
                    quat = params[start_idx:start_idx+4]
                    t = params[start_idx+4:end_idx]
                    
                    # Convert quaternion to rotation matrix
                    r = Rotation.from_quat(quat)
                    R = r.as_matrix()
                    
                    # Construct the pose
                    new_pose = np.eye(4)
                    new_pose[:3, :3] = R
                    new_pose[:3, 3] = t
                    
                    poses[i] = new_pose
                    
        # Reconstruct landmarks
        landmarks = {}
        for landmark_id, landmark in original_landmarks.items():
            if landmark_id in param_indices['landmarks']:
                start_idx, end_idx = param_indices['landmarks'][landmark_id]
                
                # Extract 3D point
                p = params[start_idx:end_idx]
                
                # Create a new landmark with the optimized position
                new_landmark = landmark.__class__(p, landmark.des, landmark.t_first)
                new_landmark.t_latest = landmark.t_latest
                new_landmark.observed_count = landmark.observed_count
                
                landmarks[landmark_id] = new_landmark
                
        return poses, landmarks
        
    def _build_jacobian_sparsity(self, param_indices, observations, landmarks, fixed_first=True):
        """
        Build the Jacobian sparsity structure for the bundle adjustment problem.
        
        Args:
            param_indices (dict): Parameter indices
            observations (dict): Observations
            landmarks (dict): Landmarks
            fixed_first (bool): Whether the first pose is fixed
            
        Returns:
            scipy.sparse.lil_matrix: Jacobian sparsity matrix
        """
        # Number of parameters
        n_params = sum(end - start for (start, end) in param_indices['poses'].values())
        n_params += sum(end - start for (start, end) in param_indices['landmarks'].values())
        
        # Number of observations (each observation contributes 2 residuals: x and y)
        n_residuals = len(observations) * 2
        
        # Create the sparsity matrix
        jac_sparsity = sparse.lil_matrix((n_residuals, n_params), dtype=int)
        
        # Fill the sparsity matrix
        residual_idx = 0
        for (frame_idx, landmark_id), _ in observations.items():
            if frame_idx > 0 or not fixed_first:  # Skip the first pose if it's fixed
                if frame_idx in param_indices['poses'] and landmark_id in param_indices['landmarks']:
                    # Get parameter indices
                    pose_start, pose_end = param_indices['poses'][frame_idx]
                    landmark_start, landmark_end = param_indices['landmarks'][landmark_id]
                    
                    # Each observation depends on the corresponding pose and landmark
                    jac_sparsity[residual_idx:residual_idx+2, pose_start:pose_end] = 1
                    jac_sparsity[residual_idx:residual_idx+2, landmark_start:landmark_end] = 1
                    
            residual_idx += 2
            
        return jac_sparsity
        
    def _project_point(self, point, pose, K):
        """
        Project a 3D point into the image plane.
        
        Args:
            point (numpy.ndarray): 3D point in world coordinates
            pose (numpy.ndarray): Camera pose (4x4)
            K (numpy.ndarray): Camera intrinsic matrix (3x3)
            
        Returns:
            numpy.ndarray: 2D point in image coordinates
        """
        # Convert to homogeneous coordinates
        point_homo = np.append(point, 1)
        
        # Transform to camera coordinates
        point_cam = pose @ point_homo
        
        # Project to image plane
        point_img = K @ point_cam[:3]
        
        # Dehomogenize
        point_img = point_img[:2] / point_img[2]
        
        return point_img