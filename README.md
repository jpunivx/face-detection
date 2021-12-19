# Face Detection
This facial detection program uses the Viola and Jones Object Detection using OpenCV, Histogram of Oriented Gradients (HOG) using the DLib library, and a Convolutional Neural Network (CNN) using the DLib library. The following are the default **Golden Images** that will be used in the program.  

   <img src="https://github.com/jpunivx/face-detection/blob/main/py-env/py-code/images/golden_images.jpg" alt="Golden Images" width="800"/>

Any image found in the [Golden Images](https://github.com/jpunivx/face-detection/tree/main/py-env/py-code/images/golden-image) directory will be used as a default Golden Image. 

Results can be found [here](https://github.com/jpunivx/face-detection/tree/main/py-env/py-code/images/result-images)

## Program Cycle
1. The program provides options to chose one of the three Algorithm
2. The program requests the user to chose to use images or a live camera feed
    * If images are selected, program requests the user to select one of these options. **Images MUST be in jpg format!**
      * Use the default Golden Images
      * Use an image provided by the user

## Pre-Requisites 
### Operating System
  This program should only be run on **Mac OSX** or **Linux**
### Python
  This program requires Python 3 to be installed. Please do not use Python over version 3.9. Some dependencies do not support the latest version of Python.
  Python can be downloaded [here](https://www.python.org/downloads/release/python-395/).
  Once installed, the correct path needs to be set for the new python 3:
      ```unlink /usr/local/bin/python``` then
      ```sudo ln -sf /usr/local/bin/python3.9 /usr/local/bin/python```
      
## Setup
* Download the zip of this repo. Unzip the download using any zip program or by using the following command in the terminal:
      ```unzip face-detection-main.zip```
* Confirm Python version is 3.9:
      ```python --version```
      
## Starting the program
Go into the new directory created, _**'face-detection-main'**_ using the terminal:
      ```cd face-detection-main```
      
Run the following in the terminal to launch the program:
      ```source run_face_detection.sh```

### The following will occur on every launch of the program
  * Creates a Virtual Python Environment in the folder py-env, if it hasn't already been done, and then sources the new environment for use.
  * Installs the python requirements [here](https://github.com/jpunivx/face-detection/blob/main/py-env/requirements_base.txt) and [here](https://github.com/jpunivx/face-detection/blob/main/py-env/requirements.txt) using pip, if needed.
  * Launches the Face Detection program
  * Finally, deactivates the Virtual Python Environment


## While Running the Program
The following are some points to remember when running the program
* Please be patient with the program. The HOG algorithm takes longer then the Viola and Jones algorithm, and the CNN algorithm takes the longest.
* To cycle through an image, press 'Q' on the keyboard
* To close a live feed of the camera, press 'Q' on the keyboard
* Please have the terminal visible, even when an image or live feed is being displayed. The terminal may provide useful information as the program runs.
* Please observe the terminal for a prompt to appear. 
* If the final image has been displayed and 'Q' has been pressed, or 'Q' has been pressed during a live feed, please observe the terminal for a user prompt. Those windows will not close until the next OpenCV window prompt occurs or the program ends.
* To end the program, enter the '0' option to go back, and finally exit.
* The default Golden Images is 6. When cycling through images, please monitor and go through each of the 6 images, during each stage of the facial detection.
