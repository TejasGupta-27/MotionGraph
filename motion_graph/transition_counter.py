def count_and_print_transitions(motion_indices):
    # Count transitions for diagnostics
    transition_count = 0
    for i in range(len(motion_indices) - 1):
        clip_idx1, frame_idx1 = motion_indices[i]
        clip_idx2, frame_idx2 = motion_indices[i+1]
        if clip_idx1 != clip_idx2 or frame_idx2 != frame_idx1 + 1:
            transition_count += 1
            print(f"Transition at frame {i}: clip {clip_idx1} frame {frame_idx1} -> clip {clip_idx2} frame {frame_idx2}")
    print(f"Total transitions: {transition_count}")
    return transition_count

