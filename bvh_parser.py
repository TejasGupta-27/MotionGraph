import numpy as np
import bvhio

def _traverse_joints(joint, joint_list):
    joint_list.append(joint)
    for child in joint.Children:
        _traverse_joints(child, joint_list)

class BvhParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.bvh = bvhio.readAsBvh(filepath)
        self.hierarchy = bvhio.readAsHierarchy(filepath)
        self.joints = []
        _traverse_joints(self.bvh.Root, self.joints)
        self.joint_map = {j.Name: h_j for j, (h_j, _, _) in zip(self.joints, self.hierarchy.layout())}

    def get_motion_data(self, include_velocities=True):
        all_frames_data = []
        num_frames = self.bvh.FrameCount

        for i in range(num_frames):
            frame_data = []
            self.hierarchy.loadPose(i)
            
            # Get root position for normalization
            root_h_joint = self.hierarchy
            root_position = np.array([root_h_joint.PositionWorld.x, root_h_joint.PositionWorld.y, root_h_joint.PositionWorld.z])

            for joint in self.joints:
                pose = joint.Keyframes[i]
                h_joint = self.joint_map[joint.Name]
                
                world_pos = h_joint.PositionWorld
                relative_pos = np.array([world_pos.x, world_pos.y, world_pos.z]) - root_position
                
                euler = pose.getEuler()
                
                for channel in joint.Channels:
                    if channel == 'Xposition':
                        frame_data.append(0.0 if joint.Name == self.bvh.Root.Name else relative_pos[0])
                    elif channel == 'Yposition':
                        if joint.Name == self.bvh.Root.Name:
                            frame_data.append(pose.Position.y) # <-- THIS IS THE FIX
                        else:
                            frame_data.append(relative_pos[1])
                    elif channel == 'Zposition':
                        frame_data.append(0.0 if joint.Name == self.bvh.Root.Name else relative_pos[2])
                    elif channel == 'Xrotation': frame_data.append(euler.x)
                    elif channel == 'Yrotation': frame_data.append(euler.y)
                    elif channel == 'Zrotation': frame_data.append(euler.z)

            all_frames_data.append(np.array(frame_data, dtype=np.float32))
        
        # Add velocity information if requested
        if include_velocities and num_frames > 1:
            velocities = []
            for i in range(num_frames):
                if i == 0:
                    # For first frame, use velocity from first to second frame
                    vel = all_frames_data[1] - all_frames_data[0]
                else:
                    # For other frames, use velocity from previous to current frame
                    vel = all_frames_data[i] - all_frames_data[i-1]
                velocities.append(vel)
            
            # Concatenate position and velocity for better matching
            all_frames_data = [np.concatenate([pos, vel]) for pos, vel in zip(all_frames_data, velocities)]
            
        return all_frames_data

if __name__ == '__main__':
    parser = BvhParser('cmu-mocap/data/001/01_01.bvh')
    motion_data = parser.get_motion_data()
    print(f"Total frames: {len(motion_data)}")
    if motion_data:
        print(f"Data shape is correct: {motion_data[0].shape[0] == sum(len(j.Channels) for j in parser.joints)}")
        print(f"First frame data shape: {motion_data[0].shape}")
