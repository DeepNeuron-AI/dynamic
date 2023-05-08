"""
Convert a video file into a collection of image files.
"""
from pathlib import Path
import math

import cv2
import pandas as pd
import numpy as np
import tqdm 


def get_num_digits(num: int) -> int:
    return math.floor(math.log10(num)) + 1


def random_frames_from_video(video: Path, output_dir: Path, desired_frames: int):
    cap = cv2.VideoCapture(str(video))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_digits = get_num_digits(num_frames)

    frame_indices = np.arange(num_frames)
    selected_indices = np.random.choice(frame_indices, desired_frames, replace=False)

    for index in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame #{index} from video ${video}")

        frame_label = str(index).zfill(num_digits)
        output_fp = output_dir / f"{video.stem}-{frame_label}.png"
        cv2.imwrite(str(output_fp), frame)

    # When everything done, release the video capture object
    cap.release()


def video_to_images(video: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_digits = get_num_digits(num_frames)

    # Read until video is completed
    frame_index = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            frame_label = str(frame_index).zfill(num_digits)
            output_fp = output_dir / f"{video.stem}-{frame_label}.png"
            cv2.imwrite(str(output_fp), frame)
        # Break the loop
        else: 
            break

        frame_index += 1

    # When everything done, release the video capture object
    cap.release()



if __name__ == "__main__":
    video_dir = Path("C:/Users/Allis/Documents/MDN/Ultrasound2023/dynamic/a4c-video-dir/Videos")
    output_dir = Path("C:/Users/Allis/Documents/MDN/Ultrasound2023/Drive_Results/echonet_images")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files =list(video_dir.iterdir())
    video_files = video_files[6982:]
    for videos in tqdm.tqdm(video_files):
        video_to_images(videos, output_dir)

    
