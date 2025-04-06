import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import DatasetLoader
from visual_odometry import VisualOdometry

def main():
    """Main entry point for the visual odometry pipeline."""
    
    # Default dataset paths
    DEFAULT_PATHS = {
        'kitti': os.path.abspath("datasets/kitti/05"),
        'malaga': os.path.abspath("datasets/malaga"),
        'parking': os.path.abspath("datasets/parking")
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Visual Odometry pipeline')
    parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'malaga', 'parking'],
                        help='Dataset to use (kitti, malaga, or parking) (default: kitti)')
    parser.add_argument('--path', type=str, help='Path to the dataset directory (if not specified, uses default path)')
    parser.add_argument('--start', type=int, default=0, help='Start frame (default: 0)')
    parser.add_argument('--end', type=int, default=-1, help='End frame (default: -1 for all frames)')
    parser.add_argument('--save', action='store_true', help='Save results to output directory')
    parser.add_argument('--no_display', action='store_true', help='Do not display visualization')
    parser.add_argument('--no_ba', action='store_true', help='Disable bundle adjustment')
    
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
    
    # Dataset-specific parameters
    param_presets = {
        'kitti': {
            "initial_frames": (0, 4),
            "forward_backward_threshold": 1.5,
            "min_bearing_angle": 3.0,
            "max_reprojection_error": 10.0,
            "keyframe_translation_threshold": 0.8,
        },
        'malaga': {
            "initial_frames": (1, 5),
            "forward_backward_threshold": 1.0,
            "min_bearing_angle": 4.0,
            "max_reprojection_error": 2.5,
            "keyframe_translation_threshold": 0.5,
        },
        'parking': {
            "initial_frames": (0, 2),
            "forward_backward_threshold": 1.2,
            "min_bearing_angle": 1.0,
            "max_reprojection_error": 3.0,
            "keyframe_translation_threshold": 0.3,
        }
    }
    
    # Get dataset-specific parameters
    params = param_presets[args.dataset.lower()]
    
    # Override parameters based on command line arguments
    params["visualize"] = not args.no_display
    params["use_bundle_adjustment"] = not args.no_ba
    
    # Create Visual Odometry instance
    vo = VisualOdometry(loader, params)
    
    # Run Visual Odometry pipeline
    try:
        print(f"Running VO pipeline on {args.dataset.upper()} dataset...")
        vo.run()
        
        # Save results if requested
        if args.save:
            vo.save_results(output_dir)
            
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