import os
import random
import numpy as np
import torchvision
from torchvision.transforms.functional import to_pil_image


def extract_equally_spaced_frames(video_path, num_frames):
    video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
    T = video.shape[0]
    if T == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    indices = np.linspace(0, T - 1, num=num_frames, dtype=int)
    frames = [video[i] for i in indices]
    pil_frames = [to_pil_image(frame.permute(2, 0, 1)) for frame in frames]
    return pil_frames


def save_frames(frames, save_dir, video_filename, prefix):
    os.makedirs(save_dir, exist_ok=True)
    base_name, _ = os.path.splitext(video_filename)
    for idx, frame in enumerate(frames):
        filename = f"{base_name}_{prefix}_frame_{idx:02d}.jpg"
        frame.save(os.path.join(save_dir, filename))


if __name__ == "__main__":
    ORIGINAL_VIDEOS_DIR = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/videos"
    FLOW_VIDEOS_DIR = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flows"
    OUTPUT_DIR = "paper_images"
    NUM_FRAMES = 10
    NUM_VIDEOS = 30

    valid_exts = (".mp4", ".avi", ".mov", ".mkv")
    original_files = [f for f in os.listdir(ORIGINAL_VIDEOS_DIR) if f.lower().endswith(valid_exts)]

    if not original_files:
        raise ValueError(f"No video files found in {ORIGINAL_VIDEOS_DIR}")

    # OPTION 1: Randomly select videos
    # for _ in range(NUM_VIDEOS):
    #     chosen_filename = random.choice(original_files)
    #     print(f"Selected video: {chosen_filename}")

    #     original_video_path = os.path.join(ORIGINAL_VIDEOS_DIR, chosen_filename)
    #     flow_video_path = os.path.join(FLOW_VIDEOS_DIR, chosen_filename)

    #     if not os.path.exists(flow_video_path):
    #         raise FileNotFoundError(f"Matching optical flow video not found: {flow_video_path}")

    #     try:
    #         original_frames = extract_equally_spaced_frames(original_video_path, NUM_FRAMES)
    #         flow_frames = extract_equally_spaced_frames(flow_video_path, NUM_FRAMES)
    #     except Exception as e:
    #         raise RuntimeError(f"Error extracting frames: {e}")

    #     original_save_dir = os.path.join(OUTPUT_DIR, chosen_filename.split(".")[0], "original")
    #     flow_save_dir = os.path.join(OUTPUT_DIR, chosen_filename.split(".")[0], "flow")

    #     save_frames(original_frames, original_save_dir, chosen_filename, prefix="original")
    #     save_frames(flow_frames, flow_save_dir, chosen_filename, prefix="flow")

    # OPTION 2: Use a predefined list of videos
    VIDEOS = [
        "AAOYRUDX.mp4",
        "ACHCJNPL.mp4",
        "ACPRKUJL.mp4",
        "ADMEMAMC.mp4",
        "ADUQOXDO.mp4",
        "AEHRLGCS.mp4",
        "AEQVOUDX.mp4",
        "AFJLVDSN.mp4",
        "AHAEZVYU.mp4",
        "AHHLBAMC.mp4",
        "AHJSFBLQ.mp4",
        "AHSAPCNX.mp4",
        "AIFSREBY.mp4",
        "AINMFXDO.mp4",
        "AJGHONSU.mp4",
        "AKCLZBQT.mp4",
        "AKQVZTHG.mp4",
        "AKYIXRSU.mp4",
        "ALYBARLL.mp4",
        "AMKXNFFP.mp4",
        "BQGQRUXS.mp4",
        "DKLTBKEW.mp4",
        "DVNMYKEW.mp4",
        "DVQBTKEW.mp4",
        "EEBXWUXS.mp4",
        "EIGMTUXS.mp4",
        "FJCHTKRF.mp4",
        "FWVWMKEW.mp4",
        "GLQUBXDO.mp4",
    ]
    for chosen_filename in VIDEOS:
        print(f"Selected video: {chosen_filename}")

        original_video_path = os.path.join(ORIGINAL_VIDEOS_DIR, chosen_filename)
        flow_video_path = os.path.join(FLOW_VIDEOS_DIR, chosen_filename)

        if not os.path.exists(flow_video_path):
            raise FileNotFoundError(f"Matching optical flow video not found: {flow_video_path}")

        try:
            original_frames = extract_equally_spaced_frames(original_video_path, NUM_FRAMES)
            flow_frames = extract_equally_spaced_frames(flow_video_path, NUM_FRAMES)
        except Exception as e:
            raise RuntimeError(f"Error extracting frames: {e}")

        original_save_dir = os.path.join(OUTPUT_DIR, chosen_filename.split(".")[0], "original")
        flow_save_dir = os.path.join(OUTPUT_DIR, chosen_filename.split(".")[0], "flow")

        save_frames(original_frames, original_save_dir, chosen_filename, prefix="original")
        save_frames(flow_frames, flow_save_dir, chosen_filename, prefix="flow")
