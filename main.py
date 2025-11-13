import time
from bvh_parser import BvhParser
from motion_graph import MotionGraph
from motion_graph.reconstruction import reconstruct_world_space_motion
from motion_graph.bvh_writer import write_bvh_file
from motion_graph.transition_counter import count_and_print_transitions

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
    motion_indices = motion_graph.generate_motion(num_frames=2500, sequential_bias=0.95)
    
    if not motion_indices:
        print("Failed to generate motion.")
        exit()

    # Reconstruction uses position-only data, which is correct
    normalized_motion_data = [motion_graph.motions_pos_only[i][j] for i, j in motion_indices]
    print(f"Generated a new motion with {len(normalized_motion_data)} frames.")
    
    # ... (Rest of the __main__ block is identical and correct) ...
    
    # Count transitions for diagnostics
    count_and_print_transitions(motion_indices)

    smoothness_score = motion_graph.calculate_smoothness(motion_indices, parser1.bvh.FrameTime)
    print(f"Smoothness score (avg velocity change): {smoothness_score:.4f}")
    
    # --- Correct World-Space Motion Reconstruction ---
    world_space_motion = reconstruct_world_space_motion(motion_graph, motion_indices, normalized_motion_data)
    
    # --- BVH Writing Logic ---
    template_bvh = parser1.bvh
    write_bvh_file(template_bvh, world_space_motion, 'generated_motion.bvh')

