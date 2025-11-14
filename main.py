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
    
    # Threshold values to test
    threshold_values = [2, 2.5, 3]
    
    # Store results for logging
    results = []
    
    print("=" * 80)
    print("Testing multiple threshold values for optimal smoothness")
    print("=" * 80)
    
    for threshold in threshold_values:
        print(f"\n{'='*80}")
        print(f"Testing threshold: {threshold}")
        print(f"{'='*80}")
        
        # Build graph with current threshold
        build_start = time.time()
        motion_graph.build_graph(threshold=threshold)
        build_end = time.time()
        build_time = build_end - build_start
        
        num_transitions_in_graph = sum(len(v) for v in motion_graph.graph.values())
        print(f"Graph built in {build_time:.2f} seconds")
        print(f"Found {num_transitions_in_graph} transitions in graph.")
        
        if num_transitions_in_graph == 0:
            print(f"WARNING: No transitions found for threshold {threshold}. Skipping...")
            results.append({
                'threshold': threshold,
                'build_time': build_time,
                'num_transitions_in_graph': num_transitions_in_graph,
                'status': 'FAILED - No transitions',
                'smoothness_score': None,
                'num_transitions_in_motion': None,
                'bvh_file': None
            })
            continue
        
        # Generate motion with high sequential bias to maintain natural walking
        # Enable blended transitions (paper approach) for better quality
        motion_indices = motion_graph.generate_motion(
            num_frames=2500, 
            sequential_bias=0.98,
            use_blended_transitions=True,
            transition_length=8  # Reduced from 10 for performance with limited data
        )
        
        if not motion_indices:
            print(f"WARNING: Failed to generate motion for threshold {threshold}. Skipping...")
            results.append({
                'threshold': threshold,
                'build_time': build_time,
                'num_transitions_in_graph': num_transitions_in_graph,
                'status': 'FAILED - Motion generation failed',
                'smoothness_score': None,
                'num_transitions_in_motion': None,
                'bvh_file': None
            })
            continue
        
        # Reconstruction uses position-only data, which is correct
        # Filter out None entries (transition placeholders) for getting source data
        valid_indices = [idx for idx in motion_indices if idx is not None and isinstance(idx, tuple)]
        normalized_motion_data = [motion_graph.motions_pos_only[i][j] for i, j in valid_indices] if valid_indices else []
        print(f"Generated a new motion with {len(motion_indices)} frames (including transitions).")
        
        # Count transitions in generated motion
        transition_count = 0
        for i in range(len(motion_indices) - 1):
            if motion_indices[i] is None or motion_indices[i+1] is None:
                continue
            if not isinstance(motion_indices[i], tuple) or not isinstance(motion_indices[i+1], tuple):
                continue
            clip_idx1, frame_idx1 = motion_indices[i]
            clip_idx2, frame_idx2 = motion_indices[i+1]
            if clip_idx1 != clip_idx2 or frame_idx2 != frame_idx1 + 1:
                transition_count += 1
        
        print(f"Total transitions in generated motion: {transition_count}")
        
        # Calculate smoothness
        smoothness_score = motion_graph.calculate_smoothness(motion_indices, parser1.bvh.FrameTime)
        print(f"Smoothness score (avg velocity change): {smoothness_score:.4f}")
        
        # --- Correct World-Space Motion Reconstruction ---
        world_space_motion = reconstruct_world_space_motion(motion_graph, motion_indices, normalized_motion_data)
        
        # --- BVH Writing Logic ---
        template_bvh = parser1.bvh
        # Replace decimal point with underscore for filename compatibility
        threshold_str = str(threshold).replace('.', '_')
        bvh_filename = f'generated_motion_threshold_{threshold_str}.bvh'
        write_bvh_file(template_bvh, world_space_motion, bvh_filename)
        
        # Store results
        results.append({
            'threshold': threshold,
            'build_time': build_time,
            'num_transitions_in_graph': num_transitions_in_graph,
            'num_transitions_in_motion': transition_count,
            'smoothness_score': smoothness_score,
            'bvh_file': bvh_filename,
            'status': 'SUCCESS'
        })
    
    # Write results to logs file
    print(f"\n{'='*80}")
    print("Writing results to logs file...")
    print(f"{'='*80}")
    
    with open('threshold_test_logs.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THRESHOLD TESTING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tested threshold values: {threshold_values}\n")
        f.write(f"Total tests: {len(threshold_values)}\n")
        f.write(f"Sequential bias: 0.98\n")
        f.write(f"Number of frames: 2500\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Write detailed results
        for result in results:
            f.write(f"Threshold: {result['threshold']}\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Build time: {result['build_time']:.2f} seconds\n")
            f.write(f"  Transitions in graph: {result['num_transitions_in_graph']}\n")
            if result['status'] == 'SUCCESS':
                f.write(f"  Transitions in motion: {result['num_transitions_in_motion']}\n")
                f.write(f"  Smoothness score: {result['smoothness_score']:.4f}\n")
                f.write(f"  BVH file: {result['bvh_file']}\n")
            f.write("\n")
        
        # Find best threshold (lowest smoothness score = best)
        successful_results = [r for r in results if r['status'] == 'SUCCESS' and r['smoothness_score'] is not None]
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['smoothness_score'])
            f.write("=" * 80 + "\n")
            f.write("BEST THRESHOLD (Lowest Smoothness Score)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Threshold: {best_result['threshold']}\n")
            f.write(f"Smoothness score: {best_result['smoothness_score']:.4f}\n")
            f.write(f"Transitions in graph: {best_result['num_transitions_in_graph']}\n")
            f.write(f"Transitions in motion: {best_result['num_transitions_in_motion']}\n")
            f.write(f"BVH file: {best_result['bvh_file']}\n")
            f.write("=" * 80 + "\n")
    
    print("Results saved to threshold_test_logs.txt")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    successful_results = [r for r in results if r['status'] == 'SUCCESS' and r['smoothness_score'] is not None]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['smoothness_score'])
        print(f"Best threshold: {best_result['threshold']} (Smoothness: {best_result['smoothness_score']:.4f})")
        print(f"BVH file: {best_result['bvh_file']}")
    else:
        print("No successful results to compare.")
    
    print(f"\nAll results saved to: threshold_test_logs.txt")
    print(f"Generated BVH files: {len([r for r in results if r['bvh_file'] is not None])} files")

