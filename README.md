# tiliter-video

This repository contains my solution for the tiliter video challenge

## Environment

Create a virtual environment for this project:

```bash
conda create -n tiliter-video python=3.6
```

Freeze the enviroment to requirements file:

```bash
pip freeze > requirements.txt
```

## Build

To restore the environment from requirements file

```bash
pip install -r requirements.txt
```

## Run

### Video processing

To run the video processing exercise:

```bash
python video.py
```

### GUI

To run the GUI:

```bash
# Segment video 1
python gui.py --video-file-path data/video_1.mp4 --segment

# Convert video 2 to monochrome
python gui.py --video-file-path data/video_2.mp4 --monochrome
```
