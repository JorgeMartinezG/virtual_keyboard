# virtual_keyboard
Virtual keyboard using different computer vision algorithms and opencv

### Intallation

Install opencv with pip

	pip install opencv-python

Alternativate using apt-get (Ubuntu)

	sudo apt-get install python-opencv

### Usage

Run camera calibration

	python virtual_keyboard.py --calibrate

Run software

	python virtual_keyboard.py

To use a different threshold value

	python virtual_keyboard.py -t <threshold_value>