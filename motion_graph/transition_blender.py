import numpy as np
import glm

def euler_to_quaternion(euler):
    """Convert Euler angles (in radians) to quaternion (x, y, z, w)."""
    # Convert to radians if needed (assuming degrees)
    euler_rad = np.radians(euler)
    
    # Roll (X), Pitch (Y), Yaw (Z)
    roll, pitch, yaw = euler_rad[0], euler_rad[1], euler_rad[2]
    
    # Quaternion conversion
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qx, qy, qz, qw])

def quaternion_to_euler(q):
    """Convert quaternion (x, y, z, w) to Euler angles (in degrees)."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    
    # Roll (X)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (Y)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (Z)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees(np.array([roll, pitch, yaw]))

def quaternion_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, negate one quaternion for shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate angle
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    # SLERP formula
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2

def blend_weight(p, k):
    """Calculate C1 continuous blend weight as per paper.
    α(p) = 2((p+1)/k)³ - 3((p+1)/k)² + 1, for -1 < p < k
    """
    if p <= -1:
        return 1.0
    if p >= k:
        return 0.0
    
    t = (p + 1) / k
    return 2 * t**3 - 3 * t**2 + 1

def create_transition_frame(parser1, frame_idx1, parser2, frame_idx2, p, k, 
                           num_channels_per_joint, root_channels):
    """Create a single blended transition frame.
    
    Args:
        parser1: Source parser
        frame_idx1: Source frame index
        parser2: Target parser  
        frame_idx2: Target frame index
        p: Transition frame index (0 to k-1)
        k: Total transition length
        num_channels_per_joint: Number of channels per joint (typically 3 for rotations)
        root_channels: Number of root channels (typically 6: Xpos, Ypos, Zpos, Xrot, Yrot, Zrot)
    
    Returns:
        Blended frame data as numpy array
    """
    # Get blend weight
    alpha = blend_weight(p, k)
    
    # Load poses
    parser1.hierarchy.loadPose(frame_idx1)
    parser2.hierarchy.loadPose(frame_idx2)
    
    # Get root positions
    root1 = parser1.bvh.Root.Keyframes[frame_idx1]
    root2 = parser2.bvh.Root.Keyframes[frame_idx2]
    
    # Blend root position (linear interpolation)
    root_pos_blended = (
        alpha * np.array([root1.Position.x, root1.Position.y, root1.Position.z]) +
        (1 - alpha) * np.array([root2.Position.x, root2.Position.y, root2.Position.z])
    )
    
    # Initialize blended frame data
    blended_frame = []
    
    # Add root position (X, Y, Z)
    blended_frame.extend(root_pos_blended)
    
    # Process all joints
    for joint in parser1.joints:
        pose1 = joint.Keyframes[frame_idx1]
        pose2 = joint.Keyframes[frame_idx2]
        
        euler1 = pose1.getEuler()
        euler2 = pose2.getEuler()
        
        # Convert to quaternions and blend with SLERP
        q1 = euler_to_quaternion(euler1)
        q2 = euler_to_quaternion(euler2)
        q_blended = quaternion_slerp(q1, q2, 1 - alpha)  # 1-alpha because we're blending TO target
        
        # Convert back to Euler
        euler_blended = quaternion_to_euler(q_blended)
        
        # Add rotations based on joint channels
        for channel in joint.Channels:
            if channel == 'Xposition':
                # Already handled for root
                if joint.Name != parser1.bvh.Root.Name:
                    blended_frame.append(0.0)  # Non-root positions are normalized
            elif channel == 'Yposition':
                if joint.Name == parser1.bvh.Root.Name:
                    blended_frame.append(root_pos_blended[1])
                else:
                    blended_frame.append(0.0)
            elif channel == 'Zposition':
                if joint.Name != parser1.bvh.Root.Name:
                    blended_frame.append(0.0)
            elif channel == 'Xrotation':
                blended_frame.append(euler_blended[0])
            elif channel == 'Yrotation':
                blended_frame.append(euler_blended[1])
            elif channel == 'Zrotation':
                blended_frame.append(euler_blended[2])
    
    return np.array(blended_frame, dtype=np.float32)

def create_transition_clip(motion_graph, source_node, target_node, transition_length=10):
    """Create a blended transition clip between two nodes.
    
    Args:
        motion_graph: MotionGraph instance
        source_node: (clip_idx, frame_idx) of source
        target_node: (clip_idx, frame_idx) of target
        transition_length: Number of frames in transition (k)
    
    Returns:
        List of blended frame data arrays
    """
    clip_idx1, frame_idx1 = source_node
    clip_idx2, frame_idx2 = target_node
    
    parser1 = motion_graph.parsers[clip_idx1]
    parser2 = motion_graph.parsers[clip_idx2]
    
    # Calculate source and target frame ranges
    # Paper: blend frames i to i+k-1 with frames j-k+1 to j
    # We'll use: source frame_idx1, target frame_idx2
    # For simplicity, we'll blend frame_idx1 with frame_idx2 over k frames
    
    transition_frames = []
    
    # Get channel info
    num_channels = len(motion_graph.motions_pos_only[0][0])
    root_channels = 6  # Xpos, Ypos, Zpos, Xrot, Yrot, Zrot
    
    for p in range(transition_length):
        # Calculate which frames to blend
        # Source: frame_idx1 (or nearby if we want window)
        # Target: frame_idx2 (or nearby if we want window)
        src_frame = max(0, min(frame_idx1, parser1.bvh.FrameCount - 1))
        tgt_frame = max(0, min(frame_idx2, parser2.bvh.FrameCount - 1))
        
        blended_frame = create_transition_frame(
            parser1, src_frame,
            parser2, tgt_frame,
            p, transition_length,
            3, root_channels
        )
        transition_frames.append(blended_frame)
    
    return transition_frames

