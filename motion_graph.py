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
        self.motions = self._load_motions()
        self.graph = {}

    def _load_motions(self):
        motions = []
        for parser in self.parsers:
            motions.append(parser.get_motion_data())
        return motions

    def build_graph(self, threshold):
        flat_motions, self.frame_indices = [], []
        for i, motion in enumerate(self.motions):
            for j, frame in enumerate(motion):
                flat_motions.append(frame)
                self.frame_indices.append((i, j))
        
        flat_motions = np.array(flat_motions)
        kdtree = KDTree(flat_motions)
        neighbors = kdtree.query_radius(flat_motions, r=threshold)
        
        for i, frame_neighbors in enumerate(neighbors):
            self.graph[self.frame_indices[i]] = [self.frame_indices[j] for j in frame_neighbors if i != j]

    def generate_motion(self, num_frames):
        motion_indices = []
        current_frame_idx = random.choice(list(self.graph.keys()))
        motion_indices.append(current_frame_idx)
        
        for _ in range(num_frames - 1):
            if not self.graph[current_frame_idx]:
                current_frame_idx = random.choice(list(self.graph.keys()))
            else:
                current_frame_idx = random.choice(self.graph[current_frame_idx])
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
                
                # Get poses before and at the transition
                pose1_before = self.motions[clip_idx1][frame_idx1 - 1]
                pose1_at = self.motions[clip_idx1][frame_idx1]
                
                pose2_at = self.motions[clip_idx2][frame_idx2]
                
                # Calculate velocities (simple difference)
                velocity1 = (pose1_at - pose1_before) / frame_time
                
                # To calculate velocity at pose2, we need the frame before it
                # If frame_idx2 is 0, we can't calculate velocity and have to skip
                if frame_idx2 > 0:
                    pose2_before = self.motions[clip_idx2][frame_idx2 - 1]
                    velocity2 = (pose2_at - pose2_before) / frame_time
                    total_velocity_change += np.linalg.norm(velocity2 - velocity1)
                else:
                    # Can't calculate velocity for the first frame of a clip, so we count it less ideally.
                    transition_count -= 1

        if transition_count == 0:
            return 0
        
        return total_velocity_change / transition_count

if __name__ == '__main__':
    parser1 = BvhParser('cmu-mocap/data/001/01_01.bvh')
    parser2 = BvhParser('cmu-mocap/data/001/01_02.bvh')

    motion_graph = MotionGraph([parser1, parser2])
    
    start_time = time.time()
    motion_graph.build_graph(threshold=10.0)
    end_time = time.time()
    
    print(f"Graph built in {end_time - start_time:.2f} seconds")
    
    num_transitions = sum(len(v) for v in motion_graph.graph.values())
    print(f"Found {num_transitions} transitions.")

    motion_indices = motion_graph.generate_motion(num_frames=500)
    normalized_motion_data = [motion_graph.motions[i][j] for i, j in motion_indices]
    print(f"Generated a new motion with {len(normalized_motion_data)} frames.")

    smoothness_score = motion_graph.calculate_smoothness(motion_indices, parser1.bvh.FrameTime)
    print(f"Smoothness score (avg velocity change): {smoothness_score:.4f}")
    
    # --- Correct World-Space Motion Reconstruction ---
    world_space_motion = []
    
    # Get the starting position from the first frame of the generated sequence
    start_clip_idx, start_frame_idx = motion_indices[0]
    motion_graph.parsers[start_clip_idx].hierarchy.loadPose(start_frame_idx)
    current_root_pos = motion_graph.parsers[start_clip_idx].hierarchy.PositionWorld
    current_root_pos = np.array([current_root_pos.x, current_root_pos.y, current_root_pos.z])

    # Reconstruct motion frame by frame
    for i in range(len(motion_indices)):
        clip_idx, frame_idx = motion_indices[i]
        
        parser = motion_graph.parsers[clip_idx]

        if i > 0:
            # Calculate HORIZONTAL delta from the source clip
            if frame_idx > 0:
                # Load current and previous poses from the source clip to get the delta
                current_pose_pos = parser.hierarchy.loadPose(frame_idx).PositionWorld
                previous_pose_pos = parser.hierarchy.loadPose(frame_idx - 1).PositionWorld
                
                delta = np.array([
                    current_pose_pos.x - previous_pose_pos.x,
                    0, # Do not integrate Y
                    current_pose_pos.z - previous_pose_pos.z
                ])
            else:
                # First frame of a clip, no delta
                delta = np.array([0.0, 0.0, 0.0])
                
            current_root_pos += delta
        
        # Get the Y position and update the frame data
        parser.hierarchy.loadPose(frame_idx)
        current_root_pos[1] = parser.hierarchy.PositionWorld.y

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
