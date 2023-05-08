"""
Convert a video file into a collection of image files.
"""
from pathlib import Path
import math

import cv2


def get_num_digits(num: int) -> int:
    return math.floor(math.log10(num)) + 1


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
    video_fp = Path("/home/lex/data/echonet-data-tiny/train/0X1A0A263B22CCD966.avi")
    output_dir = Path("/home/lex/data/echonet-data-tiny/train/test-images")
    video_to_images(video_fp, output_dir)