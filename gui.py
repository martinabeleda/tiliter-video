import argparse
from pathlib import Path
import tkinter
from typing import Callable, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageTk


# Define some default arguments since the problem statement asks for the script to be runnable without command line
# arguments
VIDEO_PATH = "data/video_1.mp4"
FRAME_RATE = 20


class SegmentationFunctor:
    """A functor which applies some segmentation algorithm to an image"""

    def __init__(self, subtractor: cv2.BackgroundSubtractor):
        self.subtractor = subtractor

    def __call__(self, image: np.ndarray) -> np.ndarray:
        forground_mask = self.subtractor.apply(image)
        return cv2.bitwise_and(image, image, mask=forground_mask)


def monochrome_fn(image: np.ndarray) -> np.ndarray:
    """A function which converts an image to grayscale"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to three channel grayscale
    image = np.stack((image,) * 3, axis=-1)
    return image


# A function which converts an image to RGB
rgb_fn = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class VideoProcessor:
    """Take a source video, apply some set of processing to it and store locally"""

    def __init__(
        self,
        video_source: str,
        processors: List[Callable],
        video_output_path: str,
        output_resolution: Optional[Tuple[int, int]],
        output_frame_rate: int = 20,
    ):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.processors = processors

        # Get video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Configure the video output
        if output_resolution is None:
            output_resolution = (self.width, self.height)
        self.out = cv2.VideoWriter(
            video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_frame_rate, output_resolution
        )

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                for fn in self.processors:  # Apply the chained processing steps to the raw frame
                    frame = fn(frame)
                return (ret, frame)
        return (False, None)

    def set_frame_back(self, num_frames: int = 1):
        current_frame = self.vid.get(cv2.CAP_PROP_POS_FRAMES) - 1
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, current_frame - num_frames)


class App:
    def __init__(self, window: tkinter.Tk, window_title: str, video_processor: VideoProcessor):
        self.window = window
        self.window.title(window_title)

        self.vid = video_processor

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.pause_button = tkinter.Button(window, text="Pause", width=50, command=self.pause)
        self.pause_button.pack(anchor=tkinter.CENTER, expand=True)

        self.step_back_button = tkinter.Button(window, text="Step Back", width=50, command=self.step_back)
        self.step_back_button["state"] = "disabled"
        self.step_back_button.pack(anchor=tkinter.CENTER, expand=True)

        self.paused = False
        self.play = None

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.configure(text="Play")
            self.step_back_button["state"] = "normal"
            self.play = self.window.after_cancel(self.play)
        else:
            # Continue updating the image
            self.pause_button.configure(text="Pause")
            self.step_back_button["state"] = "disabled"
            self.play = self.window.after(self.delay, self.update)

    def step_back(self):
        if self.paused:
            self.vid.set_frame_back()
            self.show_frame()

    def show_frame(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_fn(frame)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            return frame
        else:
            self.window.destroy()
            return None

    def update(self):
        frame = self.show_frame()
        self.vid.out.write(frame)
        self.play = self.window.after(self.delay, self.update)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file-path", help="The filepath of the video to display", type=Path, default=VIDEO_PATH)
    parser.add_argument("--target-file-path", help="The filepath of the video to extract", type=Path, default=None)
    parser.add_argument("--frame-rate", help="The target frame rate", type=int, default=FRAME_RATE)
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

    if args.target_file_path is not None:
        target_file_path = args.target_file_path
    else:
        target_file_path = args.video_file_path.resolve().parent / (str(args.video_file_path.stem) + "_processed.mp4")

    # Based on the input arguments, let's chain the processing functions. Processors could contain some arbitrary
    # number of operations to apply on the image
    processors = []
    if args.monochrome:
        processors.append(monochrome_fn)
    if args.segment:
        # Create a segmentation function using standard opencv function. Note that this has not been optimised in any
        # way but in the future, could replace this with a deep neural network.
        processors.append(SegmentationFunctor(cv2.createBackgroundSubtractorMOG2()))

    video_processor = VideoProcessor(
        video_source=str(args.video_file_path),
        processors=processors,
        video_output_path=str(target_file_path),
        output_resolution=args.output_resolution,
        output_frame_rate=args.frame_rate,
    )

    # Run the application
    App(
        window=tkinter.Tk(),
        window_title="Video Playback GUI",
        video_processor=video_processor,
    )


if __name__ == "__main__":
    main()
