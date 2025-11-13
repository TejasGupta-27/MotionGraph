import bvhio
import glm
from bvh_parser import _traverse_joints

def write_bvh_file(template_bvh, world_space_motion, output_filename):
    # --- BVH Writing Logic ---
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

    bvhio.writeBvh(output_filename, template_bvh)
    print(f"Saved generated motion to {output_filename}")

