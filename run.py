import sys
import argparse
from typing import Tuple
from pathlib import Path

import cv2


# Define some default arguments since the problem statement asks for the script to be runnable without command line
# arguments
VIDEO_PATH = "data/video_1.mp4"
FRAME_RATE = 20


def get_input() -> bool:
    """Checks for a keypress"""
    return cv2.waitKey(100) & 0xFF


def check_quit(input_key: str):
    if input_key == ord("q"):
        print("Quitting")
        sys.exit(0)


def show_frame(args: argparse.Namespace, cap, subtractor):
    ret, frame = cap.read()
    if not ret:
        sys.exit(0)

    if args.monochrome:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if args.segment:
        forground_mask = subtractor.apply(frame)
        frame = cv2.bitwise_and(frame, frame, mask=forground_mask)

    cv2.imshow("video", frame)
    return frame


def set_frame_back(cap, num_frames: int = 1):
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - num_frames)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file-path", help="The filepath of the video to display", type=Path, default=VIDEO_PATH)
    parser.add_argument("--target-file-path", help="The filepath of the video to extract", type=Path, default=None)
    parser.add_argument("--frame-rate", help="The target frame rate", type=int, default=FRAME_RATE)
    parser.add_argument("--display-resolution", help="The resolution of the video", type=Tuple[int, int], default=None)
    parser.add_argument(
        "--output-resolution", help="The resolution of the output file", type=Tuple[int, int], default=None
    )
    parser.add_argument("--monochrome", help="If set, play the video in monochrome", action="store_true")
    parser.add_argument(
        "--segment",
        help="If set, run the segmentation algorithm and save the result to the target path",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        help="Set the logger level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video_file_path))
    subtractor = cv2.createBackgroundSubtractorMOG2()

    if args.target_file_path is not None:
        target_file_path = args.target_file_path
    else:
        target_file_path = args.video_file_path.resolve().parent / (str(args.video_file_path.stem) + "_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_resolution = args.output_resolution
    if output_resolution is None:
        output_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(str(target_file_path), fourcc, args.frame_rate, output_resolution)

    while cap.isOpened():
        frame = show_frame(args, cap, subtractor)
        out.write(frame)

        input_key = get_input()
        check_quit(input_key)

        if input_key == ord("p"):
            print("Paused")

            while True:
                input_key = get_input()
                check_quit(input_key)

                if input_key == ord("b"):
                    set_frame_back(cap)
                    show_frame(args, cap, subtractor)

                if input_key == ord("p"):
                    print("Unpaused")
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
