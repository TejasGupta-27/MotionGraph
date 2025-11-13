import random
import numpy as np
from sklearn.neighbors import KDTree

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

