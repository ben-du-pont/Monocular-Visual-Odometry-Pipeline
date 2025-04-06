import cv2
import numpy as np
from copy import deepcopy

class KeyPoint:
    """
    A class to represent a keypoint with tracking history.
    """
    def __init__(self, uv, des=None, t_first=0):
        """
        Initialize a keypoint.
        
        Args:
            uv (numpy.ndarray): The pixel coordinates of the keypoint (2x1 or 2)
            des (numpy.ndarray, optional): The descriptor of the keypoint
            t_first (int, optional): The frame index when the keypoint was first detected
        """
        # Make sure uv is a 2D array with shape (2,)
        if uv is not None:
            if hasattr(uv, 'shape'):
                if len(uv.shape) > 1:
                    self.uv = uv.flatten()[:2]
                else:
                    self.uv = uv.copy()[:2]
            else:
                self.uv = np.array(uv, dtype=np.float32)[:2]
            
            self.uv = self.uv.astype(np.float32)
            # Add assertion to ensure correct shape and type
            assert self.uv.shape == (2,), f"KeyPoint uv must have shape (2,), got {self.uv.shape}"
            assert self.uv.dtype == np.float32, f"KeyPoint uv must be float32, got {self.uv.dtype}"
        else:
            self.uv = None
            
        self.uv_first = self.uv.copy() if self.uv is not None else None
        self.des = des
        self.t_first = t_first
        self.t_total = 1  # Total number of frames this keypoint has been tracked
        self.uv_history = [self.uv.copy()] if self.uv is not None else []
        self.inlier_count = 0  # Number of times this keypoint has been an inlier
        self.landmark_id = None  # ID of the associated landmark
        
    def update(self, uv):
        """
        Update the keypoint with a new position.
        
        Args:
            uv (numpy.ndarray): The new pixel coordinates
        """
        # Handle different input formats
        if hasattr(uv, 'shape'):
            if len(uv.shape) > 1:
                self.uv = uv.flatten()[:2]
            else:
                self.uv = uv.copy()[:2]
        else:
            self.uv = np.array(uv, dtype=np.float32)[:2]
            
        self.uv_history.append(self.uv.copy())
        self.t_total += 1

class Landmark:
    """
    A class to represent a 3D landmark.
    """
    def __init__(self, p, des=None, t_first=0):
        """
        Initialize a landmark.
        
        Args:
            p (numpy.ndarray): The 3D coordinates of the landmark (3x1 or 3)
            des (numpy.ndarray, optional): The descriptor of the landmark
            t_first (int, optional): The frame index when the landmark was first created
        """
        # Make sure p is a 3D array with shape (3,)
        if p is not None:
            if hasattr(p, 'shape'):
                if len(p.shape) > 1:
                    self.p = p.flatten()[:3]
                else:
                    self.p = p.copy()[:3]
            else:
                self.p = np.array(p, dtype=np.float32)[:3]

            self.p = self.p.astype(np.float32)
            # Add assertion to ensure correct shape and type
            assert self.p.shape == (3,), f"Landmark p must have shape (3,), got {self.p.shape}"
            assert self.p.dtype == np.float32, f"Landmark p must be float32, got {self.p.dtype}"
        else:
            self.p = None
        
        self.des = des
        self.t_first = t_first
        self.t_latest = 0  # Last frame index when the landmark was observed
        self.observed_count = 1  # Number of times this landmark has been observed
        self.keypoints = []  # Associated keypoints
        self.reprojection_error = float('inf')  # Current reprojection error
        
    def update(self, p):
        """
        Update the landmark position.
        
        Args:
            p (numpy.ndarray): The new 3D coordinates
        """
        # Handle different input formats
        if hasattr(p, 'shape'):
            if len(p.shape) > 1:
                self.p = p.flatten()[:3]
            else:
                self.p = p.copy()[:3]
        else:
            self.p = np.array(p, dtype=np.float32)[:3]
            
        self.observed_count += 1

class FeatureManager:
    """
    A class to manage feature detection, tracking, and matching.
    """
    def __init__(self, max_features=1000, quality_level=0.01, min_distance=10):
        """
        Initialize the feature manager.
        
        Args:
            max_features (int, optional): Maximum number of features to detect
            quality_level (float, optional): Quality level for feature detection
            min_distance (int, optional): Minimum distance between features
        """
        # Parameters for the Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=1e-4
        )
        
        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=11
        )
        
        # Parameters for SIFT feature detection and matching
        self.use_sift = True
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.sift_ratio = 0.8
        self.matcher = cv2.BFMatcher()
        
        # Previous frame for tracking
        self.prev_frame = None
        self.prev_keypoints = None
        
    def detect_keypoints(self, frame, mask=None):
        """
        Detect keypoints in a frame.
        
        Args:
            frame (numpy.ndarray): The input frame
            mask (numpy.ndarray, optional): Mask for feature detection
            
        Returns:
            list: List of KeyPoint objects
        """
        try:
            if mask is None:
                mask = np.ones_like(frame, dtype=np.uint8) * 255
                
            if self.use_sift:
                # Detect and compute SIFT features
                cv_kps = self.sift.detect(frame, mask=mask)
                
                if cv_kps:
                    kps, des = self.sift.compute(frame, cv_kps)
                    
                    # Convert OpenCV keypoints to our format
                    keypoints = []
                    for i, kp in enumerate(kps):
                        pt = np.array([kp.pt[0], kp.pt[1]], dtype=np.float32)
                        keypoint = KeyPoint(pt, des[i] if des is not None else None)
                        keypoints.append(keypoint)
                    
                    return keypoints
                else:
                    return []
            else:
                # Use goodFeaturesToTrack
                corners = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)
                
                if corners is not None and len(corners) > 0:
                    keypoints = []
                    for corner in corners:
                        pt = corner.reshape(2)
                        keypoint = KeyPoint(pt)
                        keypoints.append(keypoint)
                        
                    return keypoints
                else:
                    return []
                    
        except Exception as e:
            print(f"Error detecting keypoints: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def track_keypoints(self, prev_frame, curr_frame, keypoints, max_error=1.0):
        """
        Track keypoints from previous frame to current frame using KLT.
        
        Args:
            prev_frame (numpy.ndarray): Previous frame
            curr_frame (numpy.ndarray): Current frame
            keypoints (list): List of KeyPoint objects to track
            max_error (float, optional): Maximum bidirectional error for valid tracks
            
        Returns:
            list: List of tracked KeyPoint objects
        """

        if not keypoints:
            return []
            
        try:
            # Extract pixel coordinates from keypoints
            pts = np.array([kp.uv for kp in keypoints], dtype=np.float32)
            
            # Ensure correct shape for OpenCV
            if len(pts.shape) != 2 or pts.shape[1] != 2:
                pts = pts.reshape(-1, 2)
            
            # Add assertion to validate input
            assert pts.shape[1] == 2, f"Points must have shape (N, 2), got {pts.shape}"
            assert pts.dtype == np.float32, "Points must be float32"
                
            # Track using KLT
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, pts, None, **self.lk_params
            )
            
            if next_pts is None:
                print("KLT tracking failed completely")
                return []
            
            # Bidirectional check (forward-backward error)
            prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                curr_frame, prev_frame, next_pts, None, **self.lk_params
            )
            
            # Calculate bidirectional error
            fb_error = np.linalg.norm(pts - prev_pts_back, axis=1)
            good_pts = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < max_error)
            
            # Update and return tracked keypoints
            tracked_keypoints = []
            for i, (kp, pt, good) in enumerate(zip(keypoints, next_pts, good_pts)):
                if good:
                    # Check if point is within image boundaries
                    h, w = curr_frame.shape[:2]
                    if 0 <= pt[0] < w and 0 <= pt[1] < h:
                        # Create a copy of the keypoint and update it
                        tracked_kp = deepcopy(kp)
                        tracked_kp.update(pt)
                        
                        # Preserve landmark ID if it exists
                        if hasattr(kp, 'landmark_id') and kp.landmark_id is not None:
                            tracked_kp.landmark_id = kp.landmark_id
                            
                        tracked_keypoints.append(tracked_kp)
            tracked_indices = [i for i, good in enumerate(good_pts) if good and 0 <= next_pts[i][0] < w and 0 <= next_pts[i][1] < h]
            return tracked_keypoints, tracked_indices

        
        except Exception as e:
            print(f"Error in keypoint tracking: {e}")
            import traceback
            traceback.print_exc()
            return [], []
        
    def match_keypoints(self, keypoints1, keypoints2):
        """
        Match keypoints between two frames based on their descriptors.
        
        Args:
            keypoints1 (list): List of KeyPoint objects from the first frame
            keypoints2 (list): List of KeyPoint objects from the second frame
            
        Returns:
            list: List of tuples (index1, index2) of matching keypoints
        """
        if not keypoints1 or not keypoints2:
            return []
            
        # Extract descriptors
        des1 = np.array([kp.des for kp in keypoints1])
        des2 = np.array([kp.des for kp in keypoints2])
        
        # Match using SIFT ratio test
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < self.sift_ratio * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
                
        return good_matches
        
    def create_candidates_mask(self, frame, keypoints, min_distance=10):
        """
        Create a mask to avoid detecting new keypoints too close to existing ones.
        
        Args:
            frame (numpy.ndarray): The input frame
            keypoints (list): List of existing KeyPoint objects
            min_distance (int): Minimum distance between keypoints
            
        Returns:
            numpy.ndarray: Binary mask where 255 indicates valid regions for new keypoints
        """
        mask = np.ones_like(frame, dtype=np.uint8) * 255
        
        for kp in keypoints:
            x, y = int(kp.uv[0]), int(kp.uv[1])
            cv2.circle(mask, (x, y), min_distance, 0, -1)
            
        return mask
    
    def visualize_tracks(self, prev_frame, curr_frame, prev_pts, curr_pts, window_name="Tracks"):
        """
        Visualize feature tracks between two frames.
        
        Args:
            prev_frame (numpy.ndarray): Previous frame
            curr_frame (numpy.ndarray): Current frame
            prev_pts (list): List of keypoints in the previous frame
            curr_pts (list): List of corresponding keypoints in the current frame
            window_name (str): Name of the display window
        """
        # Create a side-by-side visualization
        if len(prev_frame.shape) == 2:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR)
        if len(curr_frame.shape) == 2:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2BGR)
            
        h, w = prev_frame.shape[:2]
        vis = np.zeros((h, w*2, 3), dtype=np.uint8)
        vis[:, :w] = prev_frame
        vis[:, w:] = curr_frame
        
        # Draw tracks
        for kp1, kp2 in zip(prev_pts, curr_pts):
            pt1 = (int(kp1.uv[0]), int(kp1.uv[1]))
            pt2 = (int(kp2.uv[0]) + w, int(kp2.uv[1]))
            
            # Draw points and connecting line
            cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
            cv2.circle(vis, pt2, 3, (0, 255, 0), -1)
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            
        # Add text showing number of tracks
        cv2.putText(vis, f"Tracks: {len(prev_pts)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(window_name, vis)