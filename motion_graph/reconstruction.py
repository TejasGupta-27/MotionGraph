import numpy as np
import logging

logger = logging.getLogger(__name__)

def reconstruct_world_space_motion(motion_graph, motion_indices, normalized_motion_data):
    """Reconstruct world-space motion with support for blended transitions."""
    # --- Correct World-Space Motion Reconstruction ---
    world_space_motion = []
    
    # Get the starting position from the first frame of the generated sequence
    start_clip_idx, start_frame_idx = motion_indices[0]
    
    # Initialize current root position at ground level (Y=0) or use the first frame's Y
    root_joint = motion_graph.parsers[start_clip_idx].bvh.Root
    
    # Bounds checking for start frame
    max_start_frame = len(root_joint.Keyframes) - 1
    if start_frame_idx > max_start_frame:
        logger.warning(f"Start frame index {start_frame_idx} out of bounds (max: {max_start_frame}). Clamping to {max_start_frame}.")
        start_frame_idx = max_start_frame
    
    motion_graph.parsers[start_clip_idx].hierarchy.loadPose(start_frame_idx)
    current_root_pos = np.array([0.0, root_joint.Keyframes[start_frame_idx].Position.y, 0.0])

    # Check if we're using blended transitions
    use_transitions = hasattr(motion_graph, '_use_blended_transitions') and motion_graph._use_blended_transitions
    transition_clips = getattr(motion_graph, '_transition_clips', [])
    
    # Build transition frame map for quick lookup
    transition_map = {}
    for start_idx, transition_frames in transition_clips:
        for offset, frame_data in enumerate(transition_frames):
            transition_map[start_idx + 1 + offset] = frame_data
    
    # Reconstruct motion frame by frame
    frame_data_idx = 0
    for i in range(len(motion_indices)):
        # Check if this is a transition frame
        if motion_indices[i] is None or i in transition_map:
            # This is a blended transition frame
            if i in transition_map:
                frame_data = transition_map[i].copy()
                # Use the transition frame data directly
                # Update root position based on previous position
                if i > 0 and len(world_space_motion) > 0:
                    # Calculate small delta from transition
                    prev_root = world_space_motion[-1][0:3]
                    current_root = frame_data[0:3]
                    # For transitions, use the blended root position but maintain continuity
                    current_root_pos = prev_root + (current_root - prev_root) * 0.1  # Smooth transition
                else:
                    current_root_pos = frame_data[0:3]
                
                frame_data[0:3] = current_root_pos
                world_space_motion.append(frame_data)
                frame_data_idx += 1
            continue
        
        clip_idx, frame_idx = motion_indices[i]
        
        parser = motion_graph.parsers[clip_idx]
        root_joint = parser.bvh.Root
        
        # Bounds checking: ensure frame_idx is within valid range
        max_frame_idx = len(root_joint.Keyframes) - 1
        if frame_idx > max_frame_idx:
            # Clamp to the last valid frame
            logger.warning(f"Frame index {frame_idx} for clip {clip_idx} out of bounds (max: {max_frame_idx}). Clamping to {max_frame_idx}.")
            frame_idx = max_frame_idx

        if i > 0:
            # Calculate HORIZONTAL delta from the source clip
            
            # Find previous non-transition frame
            prev_i = i - 1
            while prev_i >= 0 and (motion_indices[prev_i] is None or prev_i in transition_map):
                prev_i -= 1
            
            if prev_i >= 0:
                prev_clip_idx, prev_frame_idx = motion_indices[prev_i]
                
                delta = np.array([0.0, 0.0, 0.0])
                
                # IMPROVED: Calculate delta for both sequential frames and transitions
                # This eliminates "teleporting" and creates smoother motion
                if clip_idx == prev_clip_idx and frame_idx == prev_frame_idx + 1:
                    # This is a sequential frame. Calculate delta from source.
                    # Ensure frame indices are valid
                    prev_frame_to_load = min(frame_idx - 1, max_frame_idx)
                    current_pose_pos = parser.hierarchy.loadPose(frame_idx).PositionWorld
                    previous_pose_pos = parser.hierarchy.loadPose(prev_frame_to_load).PositionWorld
                    
                    delta = np.array([
                        current_pose_pos.x - previous_pose_pos.x,
                       0, # Do not integrate Y
                        current_pose_pos.z - previous_pose_pos.z
                    ])
                else:
                    # This is a transition jump. IMPROVED: Calculate delta from the new clip's previous frame
                    # instead of teleporting, to maintain motion continuity
                    if frame_idx > 0:
                        # Calculate delta from previous frame in the new clip
                        prev_frame_to_load = min(frame_idx - 1, max_frame_idx)
                        current_pose_pos = parser.hierarchy.loadPose(frame_idx).PositionWorld
                        previous_pose_pos = parser.hierarchy.loadPose(prev_frame_to_load).PositionWorld
                        
                        delta = np.array([
                            current_pose_pos.x - previous_pose_pos.x,
                           0, # Do not integrate Y
                            current_pose_pos.z - previous_pose_pos.z
                        ])
                    # If frame_idx == 0, delta remains [0,0,0] (can't calculate from previous frame)
                
                current_root_pos += delta
        
        # Get the Y position (height) from the root joint keyframe
        current_root_pos[1] = root_joint.Keyframes[frame_idx].Position.y

        if frame_data_idx < len(normalized_motion_data):
            frame_data = normalized_motion_data[frame_data_idx].copy()
        else:
            # Fallback: use current frame data
            frame_data = np.zeros(len(normalized_motion_data[0]) if normalized_motion_data else 100)
        
        # Find the root position channels and insert the new world position
        frame_data[0:3] = current_root_pos
        world_space_motion.append(frame_data)
        frame_data_idx += 1
    
    return world_space_motion

