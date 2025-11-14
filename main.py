import time
import logging
from datetime import datetime
from bvh_parser import BvhParser
from motion_graph import MotionGraph
from motion_graph.reconstruction import reconstruct_world_space_motion
from motion_graph.bvh_writer import write_bvh_file
from motion_graph.transition_counter import count_and_print_transitions

# Configure logging to write to both file and console
log_filename = f'motion_graph_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def normalize_smoothness_scores(results):
    
    successful_results = [r for r in results if r['status'] == 'SUCCESS' and r['smoothness_score'] is not None]
    
    if len(successful_results) == 0:
        return results
    
    if len(successful_results) == 1:
        # Only one result, assign middle score
        for r in results:
            if r['status'] == 'SUCCESS' and r['smoothness_score'] is not None:
                r['normalized_smoothness'] = 5.0000
        return results
    
    # Get min and max smoothness scores
    raw_scores = [r['smoothness_score'] for r in successful_results]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    
    # Normalize to 1-10 scale
    for r in results:
        if r['status'] == 'SUCCESS' and r['smoothness_score'] is not None:
            if max_score == min_score:
                # All scores are the same
                normalized = 5.0000
            else:
                # Linear normalization: min_score -> 1.0, max_score -> 10.0
                normalized = 1.0 + (r['smoothness_score'] - min_score) * 9.0 / (max_score - min_score)
            r['normalized_smoothness'] = round(normalized, 4)
        else:
            r['normalized_smoothness'] = None
    
    return results

if __name__ == '__main__':
    logger.info(f"Starting motion graph processing - Log file: {log_filename}")
    logger.info("=" * 80)
    
    # Load BVH files
    logger.info("Loading BVH files...")
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
    logger.info("Successfully loaded 10 BVH files")

    logger.info("Creating motion graph...")
    motion_graph = MotionGraph([parser1, parser2, parser3, parser4, parser5, parser6, parser7, parser8, parser9, parser10])
    logger.info("Motion graph created successfully")
    
    # Threshold values to test
    threshold_values = [3,4,5,6,7,8,9,10]
    
    # Store results for logging
    results = []
    
    logger.info("=" * 80)
    logger.info("Testing multiple threshold values for optimal smoothness")
    logger.info("=" * 80)
    
    for threshold in threshold_values:
        logger.info("")
        logger.info("="*80)
        logger.info(f"Testing threshold: {threshold}")
        logger.info("="*80)
        
        # Build graph with current threshold
        build_start = time.time()
        logger.info("Building graph...")
        motion_graph.build_graph(threshold=threshold)
        build_end = time.time()
        build_time = build_end - build_start
        
        num_transitions_in_graph = sum(len(v) for v in motion_graph.graph.values())
        logger.info(f"Graph built in {build_time:.2f} seconds")
        logger.info(f"Found {num_transitions_in_graph} transitions in graph.")
        
        if num_transitions_in_graph == 0:
            logger.warning(f"No transitions found for threshold {threshold}. Skipping...")
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
        
        
        logger.info("Generating motion sequence...")
        motion_indices = motion_graph.generate_motion(
            num_frames=2500, 
            sequential_bias=0.98,
            use_blended_transitions=True,
            transition_length=8  # Reduced from 10 for performance with limited data
        )
        
        if not motion_indices:
            logger.warning(f"Failed to generate motion for threshold {threshold}. Skipping...")
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
        
        
        valid_indices = [idx for idx in motion_indices if idx is not None and isinstance(idx, tuple)]
        normalized_motion_data = [motion_graph.motions_pos_only[i][j] for i, j in valid_indices] if valid_indices else []
        logger.info(f"Generated a new motion with {len(motion_indices)} frames (including transitions).")
        
        # Count transitions in generated motion
        logger.info("Counting transitions in generated motion...")
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
        
        logger.info(f"Total transitions in generated motion: {transition_count}")
        
        # Calculate smoothness
        logger.info("Calculating smoothness score...")
        smoothness_score = motion_graph.calculate_smoothness(motion_indices, parser1.bvh.FrameTime)
        logger.info(f"Raw smoothness score (avg velocity change): {smoothness_score:.4f}")
        
        # --- Correct World-Space Motion Reconstruction ---
        logger.info("Reconstructing world-space motion...")
        world_space_motion = reconstruct_world_space_motion(motion_graph, motion_indices, normalized_motion_data)
        
        # --- BVH Writing Logic ---
        template_bvh = parser1.bvh
        # Replace decimal point with underscore for filename compatibility
        threshold_str = str(threshold).replace('.', '_')
        bvh_filename = f'generated_motion_threshold_{threshold_str}.bvh'
        logger.info(f"Writing BVH file: {bvh_filename}")
        write_bvh_file(template_bvh, world_space_motion, bvh_filename)
        logger.info(f"BVH file written successfully: {bvh_filename}")
        
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
    
    # Normalize smoothness scores to 1-10 scale
    logger.info("")
    logger.info("="*80)
    logger.info("Normalizing smoothness scores to 1-10 scale...")
    logger.info("="*80)
    results = normalize_smoothness_scores(results)
    
    # Log normalized scores
    for result in results:
        if result['status'] == 'SUCCESS' and result.get('normalized_smoothness') is not None:
            logger.info(f"Threshold {result['threshold']}: Raw={result['smoothness_score']:.4f}, Normalized={result['normalized_smoothness']:.4f}")
    
    # Write results to logs file
    logger.info("")
    logger.info("="*80)
    logger.info("Writing results to logs file...")
    logger.info("="*80)
    
    with open('threshold_test_logs.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THRESHOLD TESTING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tested threshold values: {threshold_values}\n")
        f.write(f"Total tests: {len(threshold_values)}\n")
        f.write(f"Sequential bias: 0.98\n")
        f.write(f"Number of frames: 2500\n\n")
        f.write("NOTE: Smoothness scores are normalized to 1-10 scale\n")
        f.write("      1.0000 = Best (smoothest motion)\n")
        f.write("     10.0000 = Worst (roughest motion)\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Write detailed results
        for result in results:
            f.write(f"Threshold: {result['threshold']}\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Build time: {result['build_time']:.2f} seconds\n")
            f.write(f"  Transitions in graph: {result['num_transitions_in_graph']}\n")
            if result['status'] == 'SUCCESS':
                f.write(f"  Transitions in motion: {result['num_transitions_in_motion']}\n")
                f.write(f"  Raw smoothness score: {result['smoothness_score']:.4f}\n")
                f.write(f"  Normalized smoothness (1-10): {result['normalized_smoothness']:.4f}\n")
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
            f.write(f"Raw smoothness score: {best_result['smoothness_score']:.4f}\n")
            f.write(f"Normalized smoothness (1-10): {best_result['normalized_smoothness']:.4f}\n")
            f.write(f"Transitions in graph: {best_result['num_transitions_in_graph']}\n")
            f.write(f"Transitions in motion: {best_result['num_transitions_in_motion']}\n")
            f.write(f"BVH file: {best_result['bvh_file']}\n")
            f.write("=" * 80 + "\n")
    
    logger.info("Results saved to threshold_test_logs.txt")
    
    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    successful_results = [r for r in results if r['status'] == 'SUCCESS' and r['smoothness_score'] is not None]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['smoothness_score'])
        logger.info(f"Best threshold: {best_result['threshold']}")
        logger.info(f"  Raw smoothness: {best_result['smoothness_score']:.4f}")
        logger.info(f"  Normalized smoothness (1-10): {best_result['normalized_smoothness']:.4f}")
        logger.info(f"  BVH file: {best_result['bvh_file']}")
        logger.info("")
        logger.info("All threshold results:")
        for r in sorted(successful_results, key=lambda x: x['threshold']):
            logger.info(f"  Threshold {r['threshold']}: Normalized={r['normalized_smoothness']:.4f} (Raw={r['smoothness_score']:.4f})")
    else:
        logger.info("No successful results to compare.")
    
    logger.info("")
    logger.info(f"All results saved to: threshold_test_logs.txt")
    logger.info(f"Generated BVH files: {len([r for r in results if r['bvh_file'] is not None])} files")
    logger.info("="*80)
    logger.info(f"Complete log saved to: {log_filename}")
    logger.info("="*80)

