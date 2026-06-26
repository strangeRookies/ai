import numpy as np
import math


def append_motion_features(base_features):
    """
    Append motion features (center_drop, velocity, torso_angle) to a sequence of keypoint features.
    
    Args:
        base_features (np.ndarray): Shape (seq_len, 51) where 51 = 17 keypoints * (x, y, conf)
                                    x and y are normalized (0.0 ~ 1.0).
    Returns:
        np.ndarray: Shape (seq_len, 54) with motion features appended.
    """
    seq_len = base_features.shape[0]
    # If the sequence is empty, return as is (but this shouldn't happen)
    if seq_len == 0:
        return base_features
        
    motion_features = np.zeros((seq_len, 3), dtype=np.float32)
    
    # YOLO Pose (COCO 17) Indices:
    # 5: LShoulder, 6: RShoulder
    # 11: LHip, 12: RHip
    
    # Extract coordinates for required keypoints across the entire sequence
    # Each keypoint takes 3 indices: x, y, conf
    def get_xy(kp_idx):
        x = base_features[:, kp_idx * 3]
        y = base_features[:, kp_idx * 3 + 1]
        conf = base_features[:, kp_idx * 3 + 2]
        return x, y, conf
        
    ls_x, ls_y, ls_conf = get_xy(5)
    rs_x, rs_y, rs_conf = get_xy(6)
    lh_x, lh_y, lh_conf = get_xy(11)
    rh_x, rh_y, rh_conf = get_xy(12)
    
    # 1. Torso Angle
    # Midpoint of shoulders and hips
    shoulder_mid_x = (ls_x + rs_x) / 2.0
    shoulder_mid_y = (ls_y + rs_y) / 2.0
    hip_mid_x = (lh_x + rh_x) / 2.0
    hip_mid_y = (lh_y + rh_y) / 2.0
    
    dx = hip_mid_x - shoulder_mid_x
    dy = hip_mid_y - shoulder_mid_y
    
    # angle = atan2(dy, dx). Normalized between 0 and 1.
    # Torso is usually vertical (dy > 0, dx ~ 0) -> angle ~ pi/2
    # If falling horizontally -> angle ~ 0 or pi
    angles = np.arctan2(dy, dx)
    # Normalize angles from [-pi, pi] to [0, 1]
    torso_angle_norm = (angles + np.pi) / (2 * np.pi)
    motion_features[:, 2] = torso_angle_norm
    
    # 2. Center Drop & Velocity
    # Velocity and Drop are calculated based on the hip midpoint (center of gravity)
    # For t = 0, motion is 0
    # For t > 0, motion = pos[t] - pos[t-1]
    
    center_drop = np.zeros(seq_len, dtype=np.float32)
    velocity = np.zeros(seq_len, dtype=np.float32)
    
    # Compute differences for t > 0
    if seq_len > 1:
        diff_y = hip_mid_y[1:] - hip_mid_y[:-1]
        diff_x = hip_mid_x[1:] - hip_mid_x[:-1]
        
        # Center drop is positive if moving down (y increases)
        center_drop[1:] = diff_y
        
        # Velocity is the euclidean distance moved between frames
        velocity[1:] = np.sqrt(diff_x**2 + diff_y**2)
        
    motion_features[:, 0] = center_drop
    motion_features[:, 1] = velocity
    
    return np.concatenate([base_features, motion_features], axis=1).astype(np.float32)
