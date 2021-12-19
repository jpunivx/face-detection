#!/../bin/python3.9

# Python Imports
import os
import time
import cv2
import sys
import dlib
import numpy as np
from imutils import face_utils
import pathlib

# Full Path of File
__CURRENT_FILE_PATH__ = pathlib.Path(__file__).parent.resolve()

# Intro Message
__INTRO_TITLE__ = "Face Detection --- researched and implemented by Joseph Pildush"
__INTRO_MSG__ = "The options below will run different face detection algorithms/techniques. The purpose of this program is " +\
                "to demonstrate the different accuracies and efficiencies of using the below Facial Detection algorithms/techniques " + \
                "on an Image, or during a Live camera feed."
__CV_USAGE__ = "Images/Live Feed will open-up in different windows. Please use the 'Q' button on the keyboard " +\
                "to close the image/camera. If on close of live feed or at the conclude of the images, the last image may " +\
                "not close. If this is the case, return to this GUI and continue use. Once the program " +\
                "has fully closed, the remaining OpenCV windows will close as well."
# Menu Constants
__FILLER__ = "FILL"
__HEADER_BORDER__ = "HEADER"
__SUB_BORDER__ = "SUB"
__INPUT_STR__ = "|> "
__OPTION_ONE__ = '1'
__OPTION_TWO__ = '2'
__OPTION_THREE__ = '3'
__OPTION_RETURN__ = '0'
__MAX_OPT_SIZE__ = 60
__MAX_BORDER_SIZE__ = 65
__MAX_TITLE_SIZE__ = __MAX_BORDER_SIZE__-2
__CLEAR_SCREEN_COUNT__ = 20

# Image
__IMAGE_ROOT_PATH__ = str(__CURRENT_FILE_PATH__) + "/images/"
__IMAGE_RESULTS_PATH__ = __IMAGE_ROOT_PATH__ + "result-images/"
__IMAGE_GOLDEN_PATH__ = __IMAGE_ROOT_PATH__ + "golden-image/"
__GREEN_BORDER__ = (0, 255, 0)
__BORDER_THICKNESS__ = 10

# Viola–Jones Algorithm
__VIOLA_JONES_RESULTS_PATH__ = __IMAGE_RESULTS_PATH__ + "viola-jones/"
__CASCADE_FACE_CLASSIFIER_PATH__ = "py-env/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml"
__CASCADE_EYE_PATH__ = "py-env/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml"
__CASCADE_SMILE_PATH__ = "py-env/lib/python3.9/site-packages/cv2/data/haarcascade_smile.xml"

# Histogram of Oriented Gradient
__HOG_RESULTS_PATH__ = __IMAGE_RESULTS_PATH__ + "histogram-oriented-gradients/"
__GRADIENT_RESULTS_PATH__ = __IMAGE_RESULTS_PATH__ + "histogram-oriented-gradients/gradients/"

# CNN using DLib
__MODELS_ROOT_PATH__ = str(__CURRENT_FILE_PATH__) + "/models/"
__MODELS_FACE_DLIB__ = "mmod_human_face_detector.dat"
__MODELS_FACE_DLIB_PATH__ = __MODELS_ROOT_PATH__ + __MODELS_FACE_DLIB__
__CNN_RESULTS_PATH__ = __IMAGE_RESULTS_PATH__ + "convolutional-neural-network/"


class SavedImage:
    """
    SavedImage Class will hold the properties of the imported images

    Arguments:
        image_path (str): String holds the path to the image
        image_name (str): String holds the name of the image
        image_tag (str): String holds the tag of the image
    """
    def __init__(self, image_path="", gray=False, image=None):
        # Split the image path into different image properties
        file_name = image_path.split('/')[-1]
        image_name = file_name.split('.')[0]
        image_type = file_name.split('.')[1]

        # Only accept 'jpg' images
        if image_type == 'jpg':
            # If class was made without an image input, do the else
            if image is None:
                if gray:
                    read_image = cv2.imread(image_path, 0)
                else:
                    read_image = cv2.imread(image_path)
            else:
                read_image = image

            # Set properties of the class
            self.image_path = image_path
            self.image_name = image_name
            self.image_tag = image_type
            self.image = read_image
        else:
            # Raise exception if initialization of class brings up an error
            raise Exception("Unable to load image - " + image_path)

    def get_file_with_tag(self):
        """
        Method will combine the name of the image and the image tag to form 'image.jpg'

        Returns:
            Full image and tag name
        """
        return self.image_name + "." + self.image_tag


def clear_screen():
    """
    Method will clear console screen
    """
    print('\n'*__CLEAR_SCREEN_COUNT__)


def cv_wait(isLive=False):
    """
    Method will wait for user to hit the 'Q' key on the keyboard

    Args:
        isLive (bool): Boolean determines if a live sequence of cv2 with the camera is occurring

    Returns:
        True if the process succeeds
    """
    # If live is selected, wait until 'Q' is pressed
    if isLive:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
    else:
        # If images is selected, while true continue unless 'Q' is pressed
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        return True


def breakDownMessage(size=0, msg=""):
    """
    Method will break down the message into the the GUI format.

    Args:
        size (int): Integer determines the size of the spacing in the GUI
        msg (str): String holds the message to be refactored and displayed
    """
    message = []
    current_pos = 0

    # Loop through the message per character
    for pos in range(0, len(msg)):
        # Set pos to equal current_pos that's set after characters reach to the limit of size.
        pos = current_pos

        # Try to jump through the message per size. If index does not exist,
        # append the full message to the return message
        try:
            # If index does not exist, append the full message to the return message
            if msg[pos + size]:
                # Traverse the size backwards, and check if that space position with the msg position equates to a space
                for space in range(size, 0, -1):
                    # If a space is found, then append that segment of the message to the array message.
                    # Set the current position to after where the amendment occurred.
                    if msg[pos + space] == ' ':
                        message.append(msg[pos:pos + space])
                        current_pos = pos + space + 1
                        break
        except IndexError:
            message.append(msg[pos:])
            break
    return message


def fillMsg(msg="", is_title=False):
    """
    Method will set string contents to either a message or a title

    Args:
        msg (str): String holds the message for the GUI
        is_title (bool): Boolean determines if a title or a message needs to be created
    """
    # If a title if required
    if is_title:
        # If the length of the message exceeds the allowed GUI stretch size
        if len(msg) > __MAX_TITLE_SIZE__:
            # Breakdown the message
            message = breakDownMessage(__MAX_TITLE_SIZE__, msg)

            # Print out the array for the title fully in the GUI format
            for string in message:
                print("| " + string.ljust(__MAX_TITLE_SIZE__) + " |")
        else:
            # Print out the title fully
            print("| " + msg.ljust(__MAX_TITLE_SIZE__) + " |")
    else:
        # If the length of the message exceeds the allowed GUI stretch size
        if len(msg) > __MAX_OPT_SIZE__:
            # Breakdown the message
            message = breakDownMessage(__MAX_OPT_SIZE__, msg)

            # Print out the array message in the GUI format
            for string in message:
                print("|".ljust(__MAX_BORDER_SIZE__-__MAX_OPT_SIZE__) + string.ljust(__MAX_OPT_SIZE__) + " |")
        else:
            # Print out the message fully
            print("|".ljust(__MAX_BORDER_SIZE__-__MAX_OPT_SIZE__) + msg.ljust(__MAX_OPT_SIZE__) + " |")


def fillBorder(border_type=""):
    """
    Method will fill in the border using a certain style

    Args:
        border_type (str): String hold the border type that should be used

    Returns:
        False if the border type is invalid
    """
    # If the border if a filler border used for white space in GUI format
    if border_type == __FILLER__:
        border = "|" + " " * __MAX_BORDER_SIZE__ + "|"
        print(border)
    # If the border if a header border used to create a header border in GUI format
    elif border_type == __HEADER_BORDER__:
        border = "|" + "=" * __MAX_BORDER_SIZE__ + "|"
        print(border)
    # If the border if a sub-header border used for border to be in a sub category of the header border in GUI format
    elif border_type == __SUB_BORDER__:
        border = "|" + "-" * __MAX_BORDER_SIZE__ + "|"
        print(border)
    else:
        return False


def fillMenu(option=0, msg=""):
    """
    Method will print menu options in GUI format

    Args:
        option (int): Integer holds the option number to display
        msg (str): String holds details of the menu option

    Return:
        False if the msg is larger then the allowed size
    """
    if len(msg) > __MAX_OPT_SIZE__:
        return False
    else:
        print("| " + str(option) + " - " + msg.ljust(__MAX_OPT_SIZE__) + "|")


def getImages(gray=False):
    """
    Method will download Golden Images, or a user selected image, to be used for Facial Detection. This images will\n
    be acquired from the file system. There is a folder in this project which holds the Golden Image for this project.

    Args:
        gray (bool): Boolean determines if the images that are acquired will be gray or colour

    Returns:
        An array which contains an array of SavedImage's
    """
    try:
        # Start a loop to repeat this menu segment
        while True:
            images = []

            # Create the menu to get images
            fillBorder(__FILLER__)
            fillBorder(__SUB_BORDER__)
            fillBorder(__FILLER__)
            fillMenu(0, "Go Back")
            fillMenu(1, "Use Default Images for Detection")
            fillMenu(2, "Provide Path to Image for Detection")
            fillBorder(__FILLER__)

            # Get user inputted menu option
            menu_option = input(__INPUT_STR__)

            # If default image selection is selected
            if menu_option == __OPTION_ONE__:
                # Get the files in the directory
                files = os.listdir(__IMAGE_GOLDEN_PATH__)
                files.sort()

                # For each file
                for file in files:
                    # Attempt to create a SavedImage. If the file is not an image, an error will be displayed to the user
                    image = SavedImage(__IMAGE_GOLDEN_PATH__ + file, gray)

                    # If the Image exists, show the image
                    cv2.imshow("Original - " + image.get_file_with_tag(), image.image)

                    # Wait for user input
                    cv_wait()

                    # Append the image to the images array
                    images.append(image)

                # Return the images array
                return images
            # If user selected to provide an image path instead
            elif menu_option == __OPTION_TWO__:
                fillBorder(__FILLER__)
                fillBorder(__FILLER__)

                # User input for the path to the image to use
                path_to_image = input("|  Please provide the path to the image: ")

                # Create a Saved Image from the user image
                image = SavedImage(path_to_image, gray)

                # Show the image to the user
                cv2.imshow("Original - " + image.get_file_with_tag(), image.image)

                # Wait for user input
                cv_wait()

                # Append the image to the images array
                images.append(image)

                # Return the images array
                return images
            # If go back is selected, break the while loop
            elif menu_option == __OPTION_RETURN__:
                return
            # If an unknown option is selected, return an error
            else:
                fillBorder(__FILLER__)
                fillBorder(__FILLER__)
                fillMsg("An incorrect option has been made! Try again...")
    except Exception as e:
        fillBorder(__FILLER__)
        fillBorder(__FILLER__)
        fillMsg("ERROR! Error message is: " + str(e))


def launchFaceImageClassifier(classifier=None):
    """
    Method will begin the Viola and Jones Object Detection algorithm/technique with the\n
    application of facial detection using the OpenCV face classifier.

    Args:
        classifier (obj): Object holds an instance of the CascadeClassifier object, in the OpenCV library.
    """
    # Get the images
    images = getImages()

    # If images exist
    if images is not None:
        # Go through each image
        for image in images:
            # Pass the image into the OpenCV face classifier
            faces = classifier.detectMultiScale(image.image, scaleFactor=1.1,
                                                minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
            # For every feature found in faces - features are detected from the original image
            for (x, y, w, h) in faces:
                # For each position with a feature, draw a green border in that vicinity
                cv2.rectangle(image.image, (x, y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

            # Show the image after the full green border has been drawn onto the image
            cv2.imshow("Viola and Jones Algorithm + Face Detection - " + image.get_file_with_tag(), image.image)

            # Wait for user input
            cv_wait()

            # Save the detected face image to the file system
            cv2.imwrite(__VIOLA_JONES_RESULTS_PATH__ + image.get_file_with_tag(), image.image)


def launchFaceLiveClassifier(classifier=None):
    """
    Method will launch the live feed with the users camera, and use the Viola and Jones Object Detection
    algorithm/technique to draw a rectangle around a detected face.

    Args:
        classifier (obj): Object holds an instance of the CascadeClassifier object, in the OpenCV library.
    """

    # Launch the camera using OpenCV
    live_camera_feed = cv2.VideoCapture(0)

    # While the user has not closer the window using 'Q'
    while True:
        # Capture the current frame from the camera
        ret, frame = live_camera_feed.read()

        # Pass the frame through an OpenCV image change - to colour
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pass the colour frame into the OpenCV face classifier
        detected_faces = classifier.detectMultiScale(rgb_frame, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        # For every feature found in faces - features are detected from the original image
        for (x, y, w, h) in detected_faces:
            # Only apply a rectangle border if the width exceeds a certain number of pixels
            if w > 250:
                # For each position with a feature, draw a green border in that vicinity
                cv2.rectangle(frame, (x, y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

        # Show the frame - the live feed with the full green border around a detected face
        cv2.imshow('Live Feed', frame)

        # Pause for the user
        if cv_wait(isLive=True):
            break;

    # After the loop release the cap object
    live_camera_feed.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def camera_or_image():
    """
    Method will create the menu that asks a user if an image should be used or a users cameras.

    Return:
        Input from the user
    """
    # Print the menu for the user
    fillBorder(__FILLER__)
    fillBorder(__SUB_BORDER__)
    fillBorder(__FILLER__)
    fillMsg(__CV_USAGE__)
    fillBorder(__FILLER__)
    fillBorder(__SUB_BORDER__)
    fillBorder(__FILLER__)
    fillMenu(0, "Go Back")
    fillMenu(1, "Detect Face in Images")
    fillMenu(2, "Detect Face in Camera")
    fillBorder(__FILLER__)

    return input(__INPUT_STR__)


def launchViolaJonesClassifier():
    """
    Method will launch the menu controller for the Viola and Jones Classifier
    """
    # Store for later, OpenCV prebuilt/taught Cascade Classifiers for a face
    cascade_face = cv2.CascadeClassifier(__PROJECT_ROOT__ + __CASCADE_FACE_CLASSIFIER_PATH__)

    while True:
        # Ask the user if images will be used or a camera
        menu_option = camera_or_image()

        # If image
        if menu_option == __OPTION_ONE__:
            launchFaceImageClassifier(cascade_face)
        # If live feed
        elif menu_option == __OPTION_TWO__:
            launchFaceLiveClassifier(cascade_face)
        # If go back
        elif menu_option == __OPTION_RETURN__:
            break
        # Else provide error
        else:
            fillBorder(__FILLER__)
            fillBorder(__FILLER__)
            fillMsg("An incorrect option has been made! Try again...")


def open_cv_window_wait(images=None, title="", saveFile=False, saveFilePath=""):
    """
    Method will be used to open an image. File can also be saved to the file system using the provided save path

    Args:
        images (list): List holds the SavedImages to be displayed
        title (str): String holds the title for the OpenCV window
        saveFile (bool): Boolean determines if the image should be saved to the file system
        saveFilePath (str): String holds the file path to save the image to
    """
    for image in images:
        # Show the current image
        cv2.imshow(title + image.get_file_with_tag(), image.image)

        # If save is selected, save the image
        if saveFile:
            cv2.imwrite(saveFilePath + image.get_file_with_tag(), image.image)

        # Wait for user input to continue
        cv_wait()


def launchFaceImageHOG():
    """
    Method will launch the Histogram of Oriented Gradients algorithm/technique using the DLib library for facial detection
    """
    # Get images
    images = getImages()

    # If images exist
    if images is not None:
        # Array to hold generated image gradients
        image_gradients = []

        # For each image
        for image in images:
            # Convert image to readable matrix to calculate the gradient
            # using Sobel Edge Detection
            img = np.float32(image.image)/255.0

            # Chose a K size of 3 to be able to get the best gradient from any image
            # Acquire the gradients for the different directions using Sobel Edge Detection
            gradX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gradY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

            # Apply the gradients into one image to get the magnitude
            magnitude, angle = cv2.cartToPolar(gradX, gradY, angleInDegrees=True)

            # Append the magnitude as an image gradient as a SavedImage
            image_gradients.append(SavedImage("/" + image.get_file_with_tag(), image=magnitude))

        # Open the images to display using OpenCV and wait for user input
        open_cv_window_wait(image_gradients, "Gradients of - ", saveFile=False)

        # Provide message for the user to explain what the gradients were.
        fillBorder(__FILLER__)
        fillBorder(__FILLER__)
        fillMsg("Those gradients will now be used to calculate the histogram of the images, " +
                  "and unilaterally provide face detection using the DLib library.")

        fillBorder(__FILLER__)
        time.sleep(3)

        # Conduct Face Detection using the above gradient method and DLib Histogram Library

        # Get an instance of the DLib face detection library
        dlib_face_detection = dlib.get_frontal_face_detector()

        # For each image
        for image in images:
            # pass the image into the face detection algorithm
            detected_faces = dlib_face_detection(image.image, 1)

            # For each detected face, enumerate the results and draw a rectangle around the face
            for (i, rect) in enumerate(detected_faces):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image.image, (x,y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

        # Show the user the image and wait for input
        open_cv_window_wait(images, "Histogram of Oriented Gradients + Face Detection - ",
                            saveFile=True, saveFilePath=__HOG_RESULTS_PATH__)

        fillBorder(__FILLER__)
        fillBorder(__FILLER__)


def launchFaceLiveHOG():
    """
    Method will launch the live feed from the users camera and user HOG algorithm from DLib for facial detection.
    """
    # Launch the users camera using OpenCV
    live_camera_feed = cv2.VideoCapture(0)

    # Get the face detector library from the DLib frontal face detector library
    dlib_face_detection = dlib.get_frontal_face_detector()

    # Whilte the life feed is active
    while True:
        # Get the current live frame from the users camera
        ret, frame = live_camera_feed.read()
        live_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pass the live frame into the dlib algorithm
        detected_faces = dlib_face_detection(live_frame)

        # For each detected face, enumerate the results and draw a rectangle around the face
        for (i, rect) in enumerate(detected_faces):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

        # Show the live frame to the user
        cv2.imshow("Live Feed", frame)

        # Wait for user input to end the live feed
        if cv_wait(isLive=True):
            break;

    # After the loop release the cap object
    live_camera_feed.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def launchHistogramOrientedGradients():
    """
    Method will display the menu to the user which requests if images should be used or live face detection.
    """
    while True:
        menu_option = camera_or_image()

        if menu_option == __OPTION_ONE__:
            launchFaceImageHOG()
        elif menu_option == __OPTION_TWO__:
            launchFaceLiveHOG()
        elif menu_option == __OPTION_RETURN__:
            break
        else:
            print("\n\nAn incorrect option has been made! Try again...")


def launchFaceImageCNN():
    """
    Method will launch a CNN using DLib library, for facial detection
    """
    # Get images
    images = getImages()

    # For each images
    if images is not None:
        # Create an instance of the DLib CNN Face Detection library
        dlib_face_detection = dlib.cnn_face_detection_model_v1(__MODELS_FACE_DLIB_PATH__)

        # For each image
        for image in images:
            # Use the Dlib library to detect faces in the images
            detected_faces = dlib_face_detection(image.image)

            # For each face found
            for (i, rect) in enumerate(detected_faces):
                # Grab the points for the rectangle
                x1_left = rect.rect.left()
                y1_top = rect.rect.top()
                x2_right = rect.rect.right()
                y2_bottom = rect.rect.bottom()

                # Draw rectangle around the found face
                cv2.rectangle(image.image, (x1_left, y1_top), (x2_right, y2_bottom), __GREEN_BORDER__, __BORDER_THICKNESS__)

        # Show images and wait for user input
        open_cv_window_wait(images, "Convolutional Neural Networks + Face Detection - ",
                            saveFile=True, saveFilePath=__CNN_RESULTS_PATH__)


def launchFaceLiveCNN():
    """
    Method will launch live face detection using the CNN DLib Face Detection library
    """
    # Start the users camera
    live_camera_feed = cv2.VideoCapture(0)
    # Create an instance of the CNN DLib Face Detection library
    dlib_face_detection = dlib.cnn_face_detection_model_v1(__MODELS_FACE_DLIB_PATH__)

    # While user does not close the live feed
    while True:
        # Grab the live frame from the camera
        ret, frame = live_camera_feed.read()
        live_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the face of users in the frame using the DLib library
        detected_faces = dlib_face_detection(live_frame)

        # For each detected face
        for (i, rect) in enumerate(detected_faces):
            # Grab the points for the rectangle
            x1_left = rect.rect.left()
            y1_top = rect.rect.top()
            x2_right = rect.rect.right()
            y2_bottom = rect.rect.bottom()

            # Draw rectangle around the found face
            cv2.rectangle(frame, (x1_left, y1_top), (x2_right, y2_bottom), __GREEN_BORDER__,
                          __BORDER_THICKNESS__)

        # Show the live frame to the user
        cv2.imshow("Live Feed", frame)

        # Wait for the users input to close the live feed
        if cv_wait(isLive=True):
            break;

    # After the loop release the cap object
    live_camera_feed.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def launchConvolutionalNeuralNetwork():
    """
    Method will display the menu to the user to chose whether to launch the CNN Face Detection\n
    using an image or a live feed.
    """
    while True:
        # Get the option from the user if images are being used or the live feed
        menu_option = camera_or_image()

        if menu_option == __OPTION_ONE__:
            launchFaceImageCNN()
        elif menu_option == __OPTION_TWO__:
            launchFaceLiveCNN()
        elif menu_option == __OPTION_RETURN__:
            break
        else:
            print("\n\nAn incorrect option has been made! Try again...")


def start():
    """
    Method will launch the program
    """
    while True:
        # Display the first screen and menu options
        clear_screen()
        fillBorder(__HEADER_BORDER__)
        fillMsg(__INTRO_TITLE__, is_title=True)
        fillBorder(__HEADER_BORDER__)
        fillMsg(__INTRO_MSG__)
        fillBorder(__SUB_BORDER__)
        fillBorder(__FILLER__)
        fillMenu(0, "Exit")
        fillMenu(1, "Viola–Jones Algorithm")
        fillMenu(2, "Histogram of the Oriented Gradients")
        fillMenu(3, "Convolutional Neural Networks (CNN)")
        fillBorder(__FILLER__)

        menu_option = input(__INPUT_STR__)

        if menu_option == __OPTION_ONE__:
            launchViolaJonesClassifier()
        elif menu_option == __OPTION_TWO__:
            launchHistogramOrientedGradients()
        elif menu_option == __OPTION_THREE__:
            launchConvolutionalNeuralNetwork()
        elif menu_option == __OPTION_RETURN__:
            break
        else:
            print("Invalid Option! Please Try Again!")


if __name__ == '__main__':
    # Launch Options Menu
    global __PROJECT_ROOT__
    # Get the root directory of where the run script was launched from. This allows the OpenCV cassifier library is able
    # to find the correct classifier.
    __PROJECT_ROOT__ = sys.argv[1] + "/"
    # Start the program
    start()
