import numpy as np

def reconstruct_world_space_motion(motion_graph, motion_indices, normalized_motion_data):
    # --- Correct World-Space Motion Reconstruction ---
    world_space_motion = []
    
    # Get the starting position from the first frame of the generated sequence
    start_clip_idx, start_frame_idx = motion_indices[0]
    motion_graph.parsers[start_clip_idx].hierarchy.loadPose(start_frame_idx)
    
    # Initialize current root position at ground level (Y=0) or use the first frame's Y
    root_joint = motion_graph.parsers[start_clip_idx].bvh.Root
    current_root_pos = np.array([0.0, root_joint.Keyframes[start_frame_idx].Position.y, 0.0])

    # Reconstruct motion frame by frame
    for i in range(len(motion_indices)):
        clip_idx, frame_idx = motion_indices[i]
        
        parser = motion_graph.parsers[clip_idx]
        root_joint = parser.bvh.Root

        if i > 0:
            # Calculate HORIZONTAL delta from the source clip
            
            # We need the previous generated frame's indices to check for sequential move
            prev_clip_idx, prev_frame_idx = motion_indices[i-1]
            
            delta = np.array([0.0, 0.0, 0.0])
            
            # Only apply delta if we are in a sequential move
            # (clip_idx == prev_clip_idx) and (frame_idx == prev_frame_idx + 1)
            # The reconstruction logic for frame_idx > 0 was correct, but we
            # can be more explicit.
            if clip_idx == prev_clip_idx and frame_idx == prev_frame_idx + 1:
                # This is a sequential frame. Calculate delta from source.
                current_pose_pos = parser.hierarchy.loadPose(frame_idx).PositionWorld
                previous_pose_pos = parser.hierarchy.loadPose(frame_idx - 1).PositionWorld
                
                delta = np.array([
                    current_pose_pos.x - previous_pose_pos.x,
                   0, # Do not integrate Y
                    current_pose_pos.z - previous_pose_pos.z
                ])
            # else:
                # This is a transition jump. The delta is 0.
                # The character "teleports" to the new pose's root position,
                # but we continue from our current_root_pos.
                # The 'delta = [0,0,0]' default handles this.
            
            current_root_pos += delta
        
        # Get the Y position (height) from the root joint keyframe
        current_root_pos[1] = root_joint.Keyframes[frame_idx].Position.y

        frame_data = normalized_motion_data[i].copy()
        
        # Find the root position channels and insert the new world position
        frame_data[0:3] = current_root_pos
        world_space_motion.append(frame_data)
    
    return world_space_motion

