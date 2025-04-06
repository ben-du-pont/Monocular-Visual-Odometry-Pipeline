import cv2
import argparse
import os
import matplotlib.pyplot as plt

from src.dataset_loader import DatasetLoader
from src.visual_odometry import VisualOdometry

# Define default dataset paths
DEFAULT_PATHS = {
    'kitti': os.path.abspath("datasets/kitti/05"),
    'malaga': os.path.abspath("datasets/malaga"),
    'parking': os.path.abspath("datasets/parking")
}

def main():
    """Main function to run the Visual Odometry pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Visual Odometry pipeline')
    parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'malaga', 'parking'],
                        help='Dataset to use (kitti, malaga, or parking) (default: kitti)')
    parser.add_argument('--path', type=str, 
                        help='Path to the dataset directory (if not specified, uses default path)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start frame (default: 0)')
    parser.add_argument('--end', type=int, default=-1,
                        help='End frame (default: -1 for all frames)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to output directory')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display visualization (useful for headless servers)')
    parser.add_argument('--feature_comparison', action='store_true',
                        help='Run feature tracker comparison')
    
    args = parser.parse_args()
    
    # If path is not specified, use default path for the dataset
    if args.path is None:
        args.path = DEFAULT_PATHS[args.dataset.lower()]
        print(f"Using default path for {args.dataset}: {args.path}")
    
    # Check if path exists
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist.")
        print(f"Available default paths:")
        for dataset, path in DEFAULT_PATHS.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  - {dataset}: {path} {exists}")
        return
    
    # Create output directory
    output_dir = f'results_{args.dataset}'
    if args.save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.dataset.upper()} dataset from {args.path}...")
    loader = DatasetLoader(args.path, dataset_type=args.dataset.lower())
    
    # Initialization parameters for each dataset
    init_frames = {
        'kitti': (0, 3),
        'malaga': (1, 5),
        'parking': (0, 2)
    }
    
    # Create Visual Odometry instance
    vo = VisualOdometry(loader)
    
    # Set initialization parameters based on dataset
    vo.params["initial_frames"] = init_frames[args.dataset.lower()]
    
    # Adjust other parameters based on dataset
    if args.dataset.lower() == 'kitti':
        # KITTI has large motion and wide baseline
        vo.params["forward_backward_threshold"] = 1.5
        vo.params["alpha_threshold"] = 3.0
        vo.params["max_reprojection_error"] = 2.0
    elif args.dataset.lower() == 'malaga':
        # Malaga has smaller motion
        vo.params["forward_backward_threshold"] = 1.0
        vo.params["alpha_threshold"] = 4.0
        vo.params["max_reprojection_error"] = 2.5
    elif args.dataset.lower() == 'parking':
        # Parking has more rotation
        vo.params["forward_backward_threshold"] = 1.2
        vo.params["alpha_threshold"] = 5.0
        vo.params["max_reprojection_error"] = 3.0
    
    # Run Visual Odometry pipeline
    try:
        print(f"Running VO pipeline on {args.dataset.upper()} dataset...")
        poses = vo.run()
        
        # Save results if requested
        if args.save:
            # Save trajectory
            trajectory_file = os.path.join(output_dir, 'trajectory.txt')
            vo.save_trajectory_to_file(trajectory_file)
            
            # Save final visualization
            plt.savefig(os.path.join(output_dir, 'trajectory_plot.png'))
            
            # Save state for debugging or further analysis
            vo.save_state(os.path.join(output_dir, 'vo_state.pkl'))
            
            print(f"Results saved to {output_dir}")
        
        # Run feature tracker comparison if requested
        if args.feature_comparison:
            from src.feature_tracker_analysis import FeatureTrackerAnalysis
            print("\nRunning feature tracker comparison...")
            tracker_analysis = FeatureTrackerAnalysis(loader)
            results = tracker_analysis.run_comparison(start_frame=args.start, num_frames=30)
            fig = tracker_analysis.plot_results()
            
            if args.save:
                fig.savefig(os.path.join(output_dir, 'feature_comparison.png'))
        
        # Display final visualization unless no_display is set
        if not args.no_display:
            plt.show(block=True)
        
    except KeyboardInterrupt:
        print("\nVO pipeline interrupted by user.")
    except Exception as e:
        print(f"\nError in VO pipeline: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("\nDone.")

if __name__ == '__main__':
    main()