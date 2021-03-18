import sys
import argparse
from typing import Tuple

import cv2


# Define some default arguments since the problem statement asks for the script to be runnable without command line
# arguments
VIDEO_PATH = "data/video_1.mp4"
DISPLAY_RESOLUTION = (1920, 1080)


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
        return

    if args.monochrome:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if args.segment:
        forground_mask = subtractor.apply(frame)
        frame = cv2.bitwise_and(frame, frame, mask=forground_mask)

    cv2.imshow("video", frame)


def set_frame_back(cap, num_frames: int = 1):
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file-path", help="The filepath of the video to display", type=str, default=VIDEO_PATH)
    parser.add_argument("--target-file-path", help="The filepath of the video to extract", type=str, default=None)
    parser.add_argument(
        "--display-resolution", help="The resolution of the video", type=Tuple[int, int], default=DISPLAY_RESOLUTION
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

    cap = cv2.VideoCapture(args.video_file_path)
    subtractor = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        show_frame(args, cap, subtractor)

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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
