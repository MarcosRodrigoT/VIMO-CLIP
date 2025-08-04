import os
import cv2
import argparse
import shutil


def compute_frame_difference(video_path, output_path):
    """
    Computes the frame difference for a single video and saves the result.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video with frame differences.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # mp4v / avc1

    # Create VideoWriter object to save the output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame of {video_path}")
        cap.release()
        return

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference
        frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)

        # Write the frame difference to the output video
        out.write(frame_diff)

        # Update the previous frame
        prev_frame_gray = curr_frame_gray

    # Release everything
    cap.release()
    out.release()
    print(f"Successfully created frame difference video: {output_path}")


def main():
    """
    Main function to parse arguments and process videos from a list.
    """
    parser = argparse.ArgumentParser(description="Compute frame differences for videos.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the RGB videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output videos.")
    parser.add_argument("--video_list_file", type=str, required=True, help="Path to a text file containing a list of video filenames to process.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the list of video files to process
    with open(args.video_list_file, 'r') as f:
        videos_to_process = [line.strip() for line in f.readlines() if line.strip()]

    # Process each video in the provided list
    for filename in videos_to_process:
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(args.input_dir, filename)

            if not os.path.exists(video_path):
                print(f"Warning: Video file not found, skipping: {video_path}")
                continue

            # Set up path for the frame difference video
            output_path = os.path.join(args.output_dir, filename)

            # Compute and save the frame difference video
            compute_frame_difference(video_path, output_path)


if __name__ == "__main__":
    main()
