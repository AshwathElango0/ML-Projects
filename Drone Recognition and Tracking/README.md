To run this project, set up a new virtual environment with Python version 3.12
Activate the vritual environment.
Install poetry using pip (Run the command 'pip install poetry').

Install the dependencies by running the command 'poetry install'.

Execute the code file 'tracker.py' using this environment.
'tracker_video.py' can be used to feed in an input video, and get an annotated output video.

# Project Mechanics
The logic for tracking is based on IOU. An object is considered to be the same across frames if it has a high IOU with its previous instances.
The drones are detected using a fine-tuned FasterRCNN.
