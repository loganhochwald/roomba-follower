# Roomba Follower

Tracks a person using your webcam and calculates their position and angle relative to the center of the screen. Designed as a vision system for a Roomba-style robot, where the calculated information will be sent to an Arduino connected to the Roomba's serial import.

The iRobot® Roomba® Serial Command Interface (SCI) Specification can be found [here](https://cdn.hackaday.io/files/1747287475562752/Roomba_SCI_manual.pdf).

## Features

- Detects humans using a pretrained SSD model
- Calculates angle and distance from screen center
- Displays live video with visual guides

## Requirements

- Python 3
- OpenCV
- Model files (included):
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000_fp16.caffemodel`

## Output

Live video with:
- Bounding box around detected face
- Center coordinates and angle
- Directional vector from origin

![Demonstration Screenshot](<Example.png>)