import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

class Visualizer:
    """
    A class for visualizing the visual odometry results.
    """
    def __init__(self, window_name="Visual Odometry"):
        """
        Initialize the visualizer.
        
        Args:
            window_name (str, optional): Name of the main window
        """
        self.window_name = window_name
        self.fig = None
        self.axes = None
        self.trajectory_full = []
        self.trajectory_recent = []
        self.landmark_counts = []
        self.landmark_history = []  # Store landmarks for last 20 frames
        
        # Setup the plot
        self._setup_plots()
        
    def _setup_plots(self):
        """
        Set up the matplotlib figure and axes with fixed sizes.
        """
        plt.ion()  # Turn on interactive mode
        
        # Create a figure with fixed size
        self.fig = plt.figure(figsize=(14, 10))
        
        # Use fixed subplot positions to ensure consistent size
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # Axis for the current frame
        self.ax_frame = self.fig.add_subplot(gs[0, 0])
        self.ax_frame.set_title("Current Image")
        self.ax_frame.axis('off')
        
        # Axis for the trajectory of last 20 frames (top view)
        self.ax_recent_traj = self.fig.add_subplot(gs[0, 1])
        self.ax_recent_traj.set_title("Trajectory of last 20 frames")
        self.ax_recent_traj.set_xlabel("X")
        self.ax_recent_traj.set_ylabel("Z")
        self.ax_recent_traj.grid(True)
        
        # Axis for the landmark count over time
        self.ax_lm_count = self.fig.add_subplot(gs[1, 0])
        self.ax_lm_count.set_title("# matched landmarks over last 20 frames")
        self.ax_lm_count.set_xlabel("Frame")
        self.ax_lm_count.set_ylabel("Count")
        self.ax_lm_count.grid(True)
        
        # Axis for the full trajectory (top view)
        self.ax_full_traj = self.fig.add_subplot(gs[1, 1])
        self.ax_full_traj.set_title("Full trajectory")
        self.ax_full_traj.set_xlabel("X")
        self.ax_full_traj.set_ylabel("Z")
        self.ax_full_traj.grid(True)
        
        # Initialize the plots
        self.frame_plot = self.ax_frame.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.recent_traj_line, = self.ax_recent_traj.plot([], [], 'g-', linewidth=2)
        self.recent_landmarks = self.ax_recent_traj.scatter([], [], c='r', s=2, alpha=0.7)
        self.lm_count_line, = self.ax_lm_count.plot([], [], 'g-')
        self.full_traj_line, = self.ax_full_traj.plot([], [], 'b-', linewidth=1)
        
        # Set initial fixed axes limits with reasonable defaults
        self._initialize_fixed_axes()
        
        # Ensure all plots have equal aspect ratio for top-down view
        self.ax_recent_traj.set_aspect('equal')
        self.ax_full_traj.set_aspect('equal')
        
        # Add a tight layout
        plt.tight_layout()
        plt.pause(0.01)
        
    def _initialize_fixed_axes(self):
        """
        Initialize axes with fixed limits that will be adjusted only occasionally.
        """
        # Set initial limits for recent trajectory plot (can be adjusted as needed)
        self.ax_recent_traj.set_xlim(-20, 20)
        self.ax_recent_traj.set_ylim(-20, 20)
        
        # Set initial limits for full trajectory plot
        self.ax_full_traj.set_xlim(-50, 50)
        self.ax_full_traj.set_ylim(-50, 50)
        
        # Set initial limits for landmark count plot
        self.ax_lm_count.set_xlim(0, 20)
        self.ax_lm_count.set_ylim(0, 200)
        
        # Store the initial ranges for auto-adjustment checks
        self.recent_traj_x_range = 40  # (-20 to 20)
        self.recent_traj_z_range = 40  # (-20 to 20)
        self.full_traj_x_range = 100   # (-50 to 50)
        self.full_traj_z_range = 100   # (-50 to 50)
        
    def update(self, frame, pose, landmarks, tracked_keypoints, candidate_keypoints, frame_idx):
        """
        Update the visualization with the latest data.
        
        Args:
            frame (numpy.ndarray): Current frame
            pose (numpy.ndarray): Current camera pose (4x4)
            landmarks (list): List of 3D landmarks
            tracked_keypoints (list): List of tracked keypoints
            candidate_keypoints (list): List of candidate keypoints
            frame_idx (int): Current frame index
        """
        try:
            # Extract camera position from pose (assuming camera-to-world)
            position = pose[:3, 3]  # Directly use translation for camera-to-world pose
            
            # Add to trajectory history
            self.trajectory_full.append(position.copy())
            self.trajectory_recent.append(position.copy())
            
            # Store current landmarks
            self.landmark_history.append(landmarks)
            
            # Keep only the last 20 positions/landmarks
            if len(self.trajectory_recent) > 20:
                self.trajectory_recent.pop(0)
                self.landmark_history.pop(0)
            
            # Update landmark count
            self.landmark_counts.append(len(tracked_keypoints))
            if len(self.landmark_counts) > 20:
                self.landmark_counts.pop(0)
            
            # Update the current frame with keypoints
            self._update_frame(frame, tracked_keypoints, candidate_keypoints)
            
            # Update recent trajectory plot
            self._update_recent_trajectory()
            
            # Update landmark count plot
            self._update_landmark_count(frame_idx)
            
            # Update full trajectory plot
            self._update_full_trajectory()
            
            # Check if we need to adjust axes limits (but not on every frame)
            if frame_idx % 20 == 0:
                self._check_and_adjust_axes_limits()
            
            # Refresh the plot - don't use draw() and pause() on every frame, it can slow things down
            if frame_idx % 3 == 0:  # Update visual every 3 frames for better performance
                self.fig.canvas.draw()
                plt.pause(0.01)
        
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
        
    def _update_frame(self, frame, tracked_keypoints, candidate_keypoints):
        """
        Update the current frame display with keypoints.
        
        Args:
            frame (numpy.ndarray): Current frame
            tracked_keypoints (list): List of tracked keypoints
            candidate_keypoints (list): List of candidate keypoints
        """
        # Convert frame to BGR if grayscale
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_rgb = frame.copy()
            
        # Draw tracked keypoints (green for landmarks)
        for kp in tracked_keypoints:
            pt = (int(kp.uv[0]), int(kp.uv[1]))
            cv2.circle(frame_rgb, pt, 3, (0, 255, 0), -1)
            
        # Draw candidate keypoints (red for candidates)
        for kp in candidate_keypoints:
            pt = (int(kp.uv[0]), int(kp.uv[1]))
            cv2.circle(frame_rgb, pt, 2, (0, 0, 255), -1)
            
        # Add text with counts
        cv2.putText(frame_rgb, f"Tracked: {len(tracked_keypoints)}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_rgb, f"Candidates: {len(candidate_keypoints)}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Update the frame plot
        self.frame_plot.set_data(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
        
    def _update_recent_trajectory(self):
        """
        Update the recent trajectory plot with landmarks from previous 20 frames.
        """
        if not self.trajectory_recent:
            return
            
        # Get recent trajectory coordinates (focus on XZ plane for top view)
        traj_x = [p[0] for p in self.trajectory_recent]
        traj_z = [p[2] for p in self.trajectory_recent]
        
        # Update the trajectory plot
        self.recent_traj_line.set_data(traj_x, traj_z)
        
        # Collect all landmarks from the last 20 frames
        landmark_x = []
        landmark_z = []
        
        for landmarks in self.landmark_history:
            for lm in landmarks:
                landmark_x.append(lm.p[0])
                landmark_z.append(lm.p[2])
        
        # Update the landmark scatter plot
        if landmark_x and landmark_z:
            self.recent_landmarks.set_offsets(np.column_stack([landmark_x, landmark_z]))
        
    def _update_landmark_count(self, frame_idx):
        """
        Update the landmark count plot.
        
        Args:
            frame_idx (int): Current frame index
        """
        if not self.landmark_counts:
            return
            
        # Calculate frame indices for the last 20 frames
        frames = list(range(frame_idx - len(self.landmark_counts) + 1, frame_idx + 1))
        
        # Update the landmark count plot
        self.lm_count_line.set_data(frames, self.landmark_counts)
        
        # Update x-axis limits only
        self.ax_lm_count.set_xlim(frames[0], frames[-1])
        
    def _update_full_trajectory(self):
        """
        Update the full trajectory plot.
        """
        if not self.trajectory_full:
            return
            
        # Get full trajectory coordinates (focus on XZ plane for top view)
        traj_x = [p[0] for p in self.trajectory_full]
        traj_z = [p[2] for p in self.trajectory_full]
        
        # Update the full trajectory plot
        self.full_traj_line.set_data(traj_x, traj_z)
        
    def _check_and_adjust_axes_limits(self):
        """
        Occasionally check if axes limits need adjustment, but maintain stability.
        Adjusts limits only when trajectory exceeds 80% of current range.
        """
        if not self.trajectory_full:
            return
            
        # Full trajectory data
        full_traj_x = [p[0] for p in self.trajectory_full]
        full_traj_z = [p[2] for p in self.trajectory_full]
        
        # Recent trajectory data
        recent_traj_x = [p[0] for p in self.trajectory_recent]
        recent_traj_z = [p[2] for p in self.trajectory_recent]
        
        # Check if recent trajectory exceeds 80% of the current view range
        recent_min_x, recent_max_x = min(recent_traj_x), max(recent_traj_x)
        recent_min_z, recent_max_z = min(recent_traj_z), max(recent_traj_z)
        
        recent_x_range = recent_max_x - recent_min_x
        recent_z_range = recent_max_z - recent_min_z
        
        # Get current axis limits
        recent_xlim = self.ax_recent_traj.get_xlim()
        recent_zlim = self.ax_recent_traj.get_ylim()
        
        # Check if we need to adjust the recent trajectory view
        if (recent_x_range > 0.8 * (recent_xlim[1] - recent_xlim[0]) or
            recent_z_range > 0.8 * (recent_zlim[1] - recent_zlim[0])):
            
            # Calculate new limits with margin
            x_margin = max(5.0, recent_x_range * 0.2)
            z_margin = max(5.0, recent_z_range * 0.2)
            
            # Set new limits
            self.ax_recent_traj.set_xlim(recent_min_x - x_margin, recent_max_x + x_margin)
            self.ax_recent_traj.set_ylim(recent_min_z - z_margin, recent_max_z + z_margin)
            
            # Update stored ranges
            self.recent_traj_x_range = (recent_max_x + x_margin) - (recent_min_x - x_margin)
            self.recent_traj_z_range = (recent_max_z + z_margin) - (recent_min_z - z_margin)
        
        # Similarly check full trajectory
        full_min_x, full_max_x = min(full_traj_x), max(full_traj_x)
        full_min_z, full_max_z = min(full_traj_z), max(full_traj_z)
        
        full_x_range = full_max_x - full_min_x
        full_z_range = full_max_z - full_min_z
        
        # Get current full trajectory axis limits
        full_xlim = self.ax_full_traj.get_xlim()
        full_zlim = self.ax_full_traj.get_ylim()
        
        # Check if we need to adjust the full trajectory view
        if (full_x_range > 0.8 * (full_xlim[1] - full_xlim[0]) or
            full_z_range > 0.8 * (full_zlim[1] - full_zlim[0])):
            
            # Calculate new limits with margin
            x_margin = max(10.0, full_x_range * 0.2)
            z_margin = max(10.0, full_z_range * 0.2)
            
            # Set new limits
            self.ax_full_traj.set_xlim(full_min_x - x_margin, full_max_x + x_margin)
            self.ax_full_traj.set_ylim(full_min_z - z_margin, full_max_z + z_margin)
            
            # Update stored ranges
            self.full_traj_x_range = (full_max_x + x_margin) - (full_min_x - x_margin)
            self.full_traj_z_range = (full_max_z + z_margin) - (full_min_z - z_margin)
            
    def save_figure(self, filename):
        """
        Save the current figure to a file.
        
        Args:
            filename (str): Output filename
        """
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        
    def save_trajectory(self, filename, poses):
        """
        Save the trajectory to a file.
        
        Args:
            filename (str): Output filename
            poses (list): List of camera poses
        """
        with open(filename, 'w') as f:
            for i, pose in enumerate(poses):
                # Extract position and orientation
                R = pose[:3, :3]
                t = pose[:3, 3]
                
                # Convert rotation to quaternion
                from scipy.spatial.transform import Rotation
                r = Rotation.from_matrix(R)
                qx, qy, qz, qw = r.as_quat()
                
                # Write to file (timestamp, tx, ty, tz, qx, qy, qz, qw)
                f.write(f"{i} {t[0]} {t[1]} {t[2]} {qx} {qy} {qz} {qw}\n")