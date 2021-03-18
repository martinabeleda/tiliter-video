import argparse
from typing import Tuple

import cv2


# Define some default arguments since the problem statement asks for the script to be runnable without command line
# arguments
VIDEO_PATH = "data/video_1.mp4"
DISPLAY_RESOLUTION = (1920, 1080)


def check_input(key: str) -> bool:
    """Checks for a keypress"""
    return cv2.waitKey(100) & 0xFF == ord(key)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file-path", help="The filepath of the video to display", type=str, default=VIDEO_PATH)
    parser.add_argument(
        "--display-resolution", help="The resolution of the video", type=Tuple[int, int], default=DISPLAY_RESOLUTION
    )
    parser.add_argument("--monochrome", help="If set, play the video in monochrome", action="store_true")
    parser.add_argument(
        "--log-level",
        help="Set the logger level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("video", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if args.monochrome else frame)

        if check_input("p"):
            print("Paused")
            while True:
                if check_input("b"):
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    print(f"\tcurrent frame: {current_frame}")
                    previous_frame = current_frame - 1
                    print(f"\tsetting to previous frame: {previous_frame}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, previous_frame)
                    cv2.imshow("video", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if args.monochrome else frame)

                if check_input("p"):
                    print("Unpaused")
                    break

        if check_input("q"):
            print("Quitting")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
