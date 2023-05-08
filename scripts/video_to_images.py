"""
Convert a video file into a collection of image files.
"""
from pathlib import Path
import math

import cv2
import pandas as pd
import numpy as np


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
    video_dir = Path("/home/lex/data/echonet-data/Videos/")
    output_dir = Path("/home/lex/data/echonet-data-tiny/")
    file_csv_fp = Path("/home/lex/data/echonet-data/FileList.csv")
    total_frames = 1200
    train_prop = 0.8
    val_prop = 0.05
    frames_per_video = 5

    output_dir.mkdir(parents=True, exist_ok=True)
    file_csv = pd.read_csv(file_csv_fp)

    # Determine raw number of frames to sample for each split
    num_train_frames = int(total_frames * train_prop)
    num_val_frames = int(total_frames * val_prop)
    num_test_frames = total_frames - num_train_frames - num_val_frames
    num_frames_by_split = (num_train_frames, num_val_frames, num_test_frames)

    for split, num_frames in zip(["TRAIN", "VAL", "TEST"], num_frames_by_split):
        # Determine how many videos we'll need in order to sample that many frames
        num_videos = math.ceil(num_frames / frames_per_video)
        print(f"{split:>5}: sampling {num_videos:>3} videos in order to get {num_frames:>4} frames")

        # Randomly sample that many videos
        split_csv = (file_csv[file_csv["Split"] == split]).sample(num_videos)
        videos = [(video_dir / filename).with_suffix(".avi") for filename in split_csv["FileName"]]
        print(videos)

        # Sample `num_frames` frames from each video
        split_output_dir = output_dir / split.lower()
        split_output_dir.mkdir(exist_ok=True)
        for i, video in enumerate(videos):
            random_frames_from_video(video, split_output_dir, desired_frames=frames_per_video)