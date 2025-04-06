import numpy as np
import cv2

def project_points(points_3d, pose, K):
    """
    Project 3D points to image coordinates.
    
    Args:
        points_3d (numpy.ndarray): 3D points (Nx3)
        pose (numpy.ndarray): Camera pose (4x4)
        K (numpy.ndarray): Camera intrinsic matrix (3x3)
        
    Returns:
        numpy.ndarray: Projected points (Nx2)
    """
    # Convert to homogeneous coordinates
    points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Transform to camera coordinates
    points_cam = (pose @ points_homo.T).T[:, :3]
    
    # Project to image plane
    points_img = (K @ points_cam.T).T
    
    # Dehomogenize
    points_img = points_img[:, :2] / points_img[:, 2:]
    
    return points_img

def compute_reprojection_error(points_3d, points_2d, pose, K):
    """
    Compute reprojection error.
    
    Args:
        points_3d (numpy.ndarray): 3D points (Nx3)
        points_2d (numpy.ndarray): 2D points (Nx2)
        pose (numpy.ndarray): Camera pose (4x4)
        K (numpy.ndarray): Camera intrinsic matrix (3x3)
        
    Returns:
        numpy.ndarray: Reprojection errors (N)
    """
    # Project 3D points
    points_proj = project_points(points_3d, pose, K)
    
    # Compute error
    errors = np.linalg.norm(points_proj - points_2d, axis=1)
    
    return errors

def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=3, thickness=1):
    """
    Draw keypoints on an image.
    
    Args:
        image (numpy.ndarray): Input image
        keypoints (list): List of KeyPoint objects
        color (tuple): BGR color
        radius (int): Circle radius
        thickness (int): Circle thickness
        
    Returns:
        numpy.ndarray: Image with keypoints
    """
    # Convert to color if grayscale
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()
        
    # Draw keypoints
    for kp in keypoints:
        pt = (int(kp.uv[0]), int(kp.uv[1]))
        cv2.circle(image_rgb, pt, radius, color, thickness)
        
    return image_rgb

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    """
    Draw matches between two images.
    
    Args:
        img1 (numpy.ndarray): First image
        keypoints1 (list): List of KeyPoint objects in the first image
        img2 (numpy.ndarray): Second image
        keypoints2 (list): List of KeyPoint objects in the second image
        matches (list): List of tuples (idx1, idx2) defining matches
        
    Returns:
        numpy.ndarray: Image with matches
    """
    # Convert to color if grayscale
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_rgb = img1.copy()
        
    if len(img2.shape) == 2:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_rgb = img2.copy()
        
    # Create output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:h1, :w1] = img1_rgb
    output[:h2, w1:] = img2_rgb
    
    # Draw matches
    for idx1, idx2 in matches:
        pt1 = (int(keypoints1[idx1].uv[0]), int(keypoints1[idx1].uv[1]))
        pt2 = (int(keypoints2[idx2].uv[0]) + w1, int(keypoints2[idx2].uv[1]))
        
        # Draw line
        cv2.line(output, pt1, pt2, (0, 255, 0), 1)
        
        # Draw points
        cv2.circle(output, pt1, 3, (0, 0, 255), -1)
        cv2.circle(output, pt2, 3, (0, 0, 255), -1)
        
    return output

def visualize_epipolar_lines(img1, img2, pts1, pts2, F):
    """
    Visualize epipolar lines.
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
        pts1 (numpy.ndarray): Points in the first image (Nx2)
        pts2 (numpy.ndarray): Points in the second image (Nx2)
        F (numpy.ndarray): Fundamental matrix (3x3)
        
    Returns:
        numpy.ndarray: Image with epipolar lines
    """
    # Convert to color if grayscale
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_rgb = img1.copy()
        
    if len(img2.shape) == 2:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_rgb = img2.copy()
        
    # Convert points to proper format
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    # Compute epipolar lines in the first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    
    # Compute epipolar lines in the second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    # Draw lines on the first image
    for i, (pt, line) in enumerate(zip(pts1, lines1)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [img1.shape[1], -(line[2] + line[0]*img1.shape[1])/line[1]])
        img1_rgb = cv2.line(img1_rgb, (x0, y0), (x1, y1), color, 1)
        img1_rgb = cv2.circle(img1_rgb, tuple(pt), 5, color, -1)
        
    # Draw lines on the second image
    for i, (pt, line) in enumerate(zip(pts2, lines2)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [img2.shape[1], -(line[2] + line[0]*img2.shape[1])/line[1]])
        img2_rgb = cv2.line(img2_rgb, (x0, y0), (x1, y1), color, 1)
        img2_rgb = cv2.circle(img2_rgb, tuple(pt), 5, color, -1)
        
    # Create output image
    output = np.hstack((img1_rgb, img2_rgb))
    
    return output