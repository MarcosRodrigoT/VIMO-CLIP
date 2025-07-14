import os
import cv2
import argparse
import numpy as np
import shutil


def compute_optical_flow(video_path, output_path):
    """
    Computes the dense optical flow for a video and saves the visualization.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output optical flow video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Create VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Read the first frame and convert to grayscale
        ret, first_frame = cap.read()
        if not ret:
            print(f"Error: Could not read the first frame of {video_path}")
            cap.release()
            return
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Create a mask for visualization
        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255  # Set saturation to maximum

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Compute magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Set image hue according to the optical flow direction
            mask[..., 0] = angle * 180 / np.pi / 2

            # Set image value according to the optical flow magnitude (normalized)
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Convert HSV to BGR color representation
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

            # Write the BGR frame to the output video
            out.write(rgb)

            # Update previous frame
            prev_gray = gray

    finally:
        cap.release()
        out.release()
        print(f"Successfully created optical flow video: {output_path}")


def main():
    """
    Main function to parse arguments and process all videos.
    """
    parser = argparse.ArgumentParser(description="Generate dense optical flow videos.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the RGB videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output flow videos.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(args.input_dir, filename)

            # Copy original video to output directory
            shutil.copy(video_path, args.output_dir)
            print(f"Copied original video: {os.path.join(args.output_dir, filename)}")

            # Set up path for the optical flow video
            output_filename = f"{os.path.splitext(filename)[0]}_output.mp4"
            output_path = os.path.join(args.output_dir, output_filename)

            # Compute and save the optical flow video
            compute_optical_flow(video_path, output_path)


if __name__ == "__main__":
    main()
