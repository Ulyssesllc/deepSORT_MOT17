# vim: expandtab:ts=4:sw=4
import argparse
import os
import deep_sort_app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deep SORT evaluation on MOT17")
    parser.add_argument(
        "--mot_dir",
        help="Path to MOT17 split directory (e.g., MOT17/train or MOT17/test)",
        required=True,
    )
    parser.add_argument(
        "--detection_dir",
        help="Path to detections.",
        default="detections",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Folder in which the results will be stored. Will "
        "be created if it does not exist.",
        default="results",
    )
    parser.add_argument(
        "--min_confidence",
        help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value. Set to "
        "0.3 to reproduce results in the paper.",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "--min_detection_height",
        help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--nms_max_overlap",
        help="Non-maximum suppression threshold: Maximum detection overlap.",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--max_cosine_distance",
        help="Gating threshold for cosine distance metric (object appearance).",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--nn_budget",
        help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.",
        type=int,
        default=100,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = [
        d
        for d in sorted(os.listdir(args.mot_dir))
        if os.path.isdir(os.path.join(args.mot_dir, d)) and not d.startswith(".")
    ]
    skipped = 0
    ran = 0
    for sequence in sequences:
        sequence_dir = os.path.join(args.mot_dir, sequence)
        detection_file = os.path.join(args.detection_dir, f"{sequence}.npy")
        if not os.path.exists(detection_file):
            print(
                f"[WARN] Missing detections for {sequence}, expected: {detection_file}. Skipping."
            )
            skipped += 1
            continue
        output_file = os.path.join(args.output_dir, f"{sequence}.txt")
        print(f"Running sequence {sequence}")
        deep_sort_app.run(
            sequence_dir,
            detection_file,
            output_file,
            args.min_confidence,
            args.nms_max_overlap,
            args.min_detection_height,
            args.max_cosine_distance,
            args.nn_budget,
            display=False,
        )
        ran += 1
    print(f"Done. Ran {ran} sequences, skipped {skipped} (missing detections).")
