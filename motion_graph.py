import time
import random
import numpy as np
from sklearn.neighbors import KDTree
import bvhio
from bvh_parser import BvhParser, _traverse_joints
import glm

class MotionGraph:
    def __init__(self, parsers):
        self.parsers = parsers
        # Load motions WITH velocity for building the graph
        self.motions = self._load_motions() 
        # Load motions WITHOUT velocity for reconstruction
        self.motions_pos_only = self._load_motions_pos_only()
        self.graph = {}
        self.frame_indices = [] # Make this a class member

    def _load_motions(self):
        # FIX 1: Load motions WITH velocity (True) for better matching
        print("Loading motions with velocity for graph...")
        motions = []
        for parser in self.parsers:
            motions.append(parser.get_motion_data(include_velocities=True))
        print("Done.")
        return motions
    
    def _load_motions_pos_only(self):
        # FIX 2: This function now correctly loads position-only data
        print("Loading motions with position-only for reconstruction...")
        motions = []
        for parser in self.parsers:
            motions.append(parser.get_motion_data(include_velocities=False))
        print("Done.")
        return motions

    def build_graph(self, threshold):
        flat_motions = []
        # Clear frame indices for rebuild
        self.frame_indices = [] 
        
        for i, motion in enumerate(self.motions):
            for j, frame in enumerate(motion):
                flat_motions.append(frame)
                self.frame_indices.append((i, j))
        
        flat_motions = np.array(flat_motions)
        if flat_motions.shape[0] == 0:
            print("Error: No motion data loaded.")
            return

        print(f"Building KDTree on {flat_motions.shape[0]} frames...")
        kdtree = KDTree(flat_motions)
        print("Querying neighbors...")
        neighbors = kdtree.query_radius(flat_motions, r=threshold)
        
        print("Building graph...")
        for i, frame_neighbors in enumerate(neighbors):
            current_node = self.frame_indices[i] # (clip_idx, frame_idx)
            
            # FIX 3: Do not allow transitions FROM the first frame of a clip
            if current_node[1] == 0: # if frame_idx == 0
                self.graph[current_node] = []
                continue

            valid_neighbors = []
            for j in frame_neighbors:
                if i == j:
                    continue
                
                neighbor_node = self.frame_indices[j]
                
                # FIX 3: Do not allow transitions TO the first frame of a clip
                if neighbor_node[1] > 0: # if frame_idx > 0
                    valid_neighbors.append(neighbor_node)
                    
            self.graph[current_node] = valid_neighbors

    def generate_motion(self, num_frames, sequential_bias=0.7):
        motion_indices = []
        
        # FIX 3: Ensure we don't start on frame 0
        valid_start_nodes = [node for node in self.graph.keys() if node[1] > 0]
        if not valid_start_nodes:
            print("Error: No valid start nodes found (all nodes are frame 0?).")
            return []
            
        current_frame_idx = random.choice(valid_start_nodes)
        motion_indices.append(current_frame_idx)
        
        for _ in range(num_frames - 1):
            clip_idx, frame_idx = current_frame_idx
            
            # Check if next sequential frame exists in the same clip
            next_sequential = (clip_idx, frame_idx + 1)
            # Use motions_pos_only for length check, as it's the one for reconstruction
            max_frame = len(self.motions_pos_only[clip_idx]) - 1 
            
            # Prefer sequential frames with a probability bias
            if frame_idx < max_frame and random.random() < sequential_bias:
                # Continue with next frame in sequence
                current_frame_idx = next_sequential
            else:
                # Make a transition using the motion graph
                possible_transitions = self.graph.get(current_frame_idx, [])
                
                if not possible_transitions:
                    # No transitions from this frame, pick a new random valid frame
                    current_frame_idx = random.choice(valid_start_nodes)
                else:
                    # Make the jump
                    current_frame_idx = random.choice(possible_transitions)
            
            motion_indices.append(current_frame_idx)
            
        return motion_indices

    def calculate_smoothness(self, motion_indices, frame_time):
        total_velocity_change = 0
        transition_count = 0

        for i in range(len(motion_indices) - 1):
            clip_idx1, frame_idx1 = motion_indices[i]
            clip_idx2, frame_idx2 = motion_indices[i+1]

            # Check if it's a transition (not just the next frame in the same clip)
            if clip_idx1 != clip_idx2 or frame_idx2 != frame_idx1 + 1:
                transition_count += 1
                
                if frame_idx1 == 0 or frame_idx2 == 0:
                     # This shouldn't happen for a transition, but if it does, skip it
                     transition_count -= 1
                     continue

                # Get poses before and at the transition (use position-only data)
                pose1_before = self.motions_pos_only[clip_idx1][frame_idx1 - 1]
                pose1_at = self.motions_pos_only[clip_idx1][frame_idx1]
                
                pose2_before = self.motions_pos_only[clip_idx2][frame_idx2 - 1]
                pose2_at = self.motions_pos_only[clip_idx2][frame_idx2]
                
                # Calculate velocities (simple difference)
                velocity1 = (pose1_at - pose1_before) / frame_time
                velocity2 = (pose2_at - pose2_before) / frame_time
                
                total_velocity_change += np.linalg.norm(velocity2 - velocity1)

        if transition_count == 0:
            return 0
        
        return total_velocity_change / transition_count

if __name__ == '__main__':
    parser1 = BvhParser('cmu-mocap/data/001/01_01.bvh')
    parser2 = BvhParser('cmu-mocap/data/001/01_02.bvh')
    parser3 = BvhParser('cmu-mocap/data/001/01_03.bvh')
    parser4 = BvhParser('cmu-mocap/data/001/01_04.bvh')
    parser5 = BvhParser('cmu-mocap/data/001/01_05.bvh')
    parser6 = BvhParser('cmu-mocap/data/001/01_06.bvh')
    parser7 = BvhParser('cmu-mocap/data/001/01_07.bvh')
    parser8 = BvhParser('cmu-mocap/data/001/01_08.bvh')
    parser9 = BvhParser('cmu-mocap/data/001/01_09.bvh')
    parser10 = BvhParser('cmu-mocap/data/001/01_10.bvh')


    motion_graph = MotionGraph([parser1, parser2, parser3, parser4, parser5, parser6, parser7, parser8, parser9, parser10])
    
    start_time = time.time()
    
    # FIX 4: The threshold MUST be larger now.
    # The feature vector includes velocity, so distances are bigger.
    # This value requires tuning!
    # - Too high = "glitchy" transitions (like you have now).
    # - Too low = very few or no transitions.
    # Start with a value like 10.0 or 15.0 and adjust.
    motion_graph.build_graph(threshold=6.0)
    
    end_time = time.time()
    
    print(f"Graph built in {end_time - start_time:.2f} seconds")
    
    num_transitions = sum(len(v) for v in motion_graph.graph.values())
    print(f"Found {num_transitions} transitions.")

    if num_transitions == 0:
        print("\n!!! WARNING: No transitions found. Try INCREASING the threshold. !!!\n")
    
    # Generate motion with high sequential bias to maintain natural walking
    motion_indices = motion_graph.generate_motion(num_frames=500, sequential_bias=0.95)
    
    if not motion_indices:
        print("Failed to generate motion.")
        exit()

    # Reconstruction uses position-only data, which is correct
    normalized_motion_data = [motion_graph.motions_pos_only[i][j] for i, j in motion_indices]
    print(f"Generated a new motion with {len(normalized_motion_data)} frames.")
    
    # ... (Rest of the __main__ block is identical and correct) ...
    
    # Count transitions for diagnostics
    transition_count = 0
    for i in range(len(motion_indices) - 1):
        clip_idx1, frame_idx1 = motion_indices[i]
        clip_idx2, frame_idx2 = motion_indices[i+1]
        if clip_idx1 != clip_idx2 or frame_idx2 != frame_idx1 + 1:
            transition_count += 1
            print(f"Transition at frame {i}: clip {clip_idx1} frame {frame_idx1} -> clip {clip_idx2} frame {frame_idx2}")
    print(f"Total transitions: {transition_count}")

    smoothness_score = motion_graph.calculate_smoothness(motion_indices, parser1.bvh.FrameTime)
    print(f"Smoothness score (avg velocity change): {smoothness_score:.4f}")
    
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
    
    # --- BVH Writing Logic ---
    template_bvh = parser1.bvh
    flat_joint_list = []
    _traverse_joints(template_bvh.Root, flat_joint_list)

    template_bvh.FrameCount = len(world_space_motion)
    for joint in flat_joint_list:
        joint.Keyframes = [bvhio.Pose() for _ in range(len(world_space_motion))]

    current_pos_in_frame = 0
    for joint_idx, joint in enumerate(flat_joint_list):
        num_channels = len(joint.Channels)
        for frame_idx in range(len(world_space_motion)):
            frame_data_slice = world_space_motion[frame_idx][current_pos_in_frame : current_pos_in_frame + num_channels]
            
            pose = joint.Keyframes[frame_idx]
            data_list = list(frame_data_slice)
            
            pos = glm.vec3(0,0,0)
            euler = glm.vec3(0,0,0)

            for channel in joint.Channels:
                if channel == 'Xposition': pos.x = data_list.pop(0)
                elif channel == 'Yposition': pos.y = data_list.pop(0)
                elif channel == 'Zposition': pos.z = data_list.pop(0)
                elif channel == 'Xrotation': euler.x = data_list.pop(0)
                elif channel == 'Yrotation': euler.y = data_list.pop(0)
                elif channel == 'Zrotation': euler.z = data_list.pop(0)

            pose.Position = pos
            pose.setEuler(euler)

        current_pos_in_frame += num_channels

    bvhio.writeBvh('generated_motion.bvh', template_bvh)
    print("Saved generated motion to generated_motion.bvh")