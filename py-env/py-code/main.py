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
__DEFAULT_IMAGE_NUM__ = 4
__IMAGE_ROOT_PATH__ = str(__CURRENT_FILE_PATH__) + "/images/"
__IMAGE_RESULTS_PATH__ = __IMAGE_ROOT_PATH__ + "result-images/"
__IMAGE_GOLDEN_PATH__ = __IMAGE_ROOT_PATH__ + "golden-image/"
__IMAGE_PERSON_FILE__ = "person"
__IMAGE_PERSON_FILE_TAG__ = ".jpg"
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
    def __init__(self, image_path="", gray=False, image=None):
        file_name = image_path.split('/')[-1]
        image_name = file_name.split('.')[0]
        image_type = file_name.split('.')[1]

        if image_type == 'jpg':
            if image is None:
                if gray:
                    read_image = cv2.imread(image_path, 0)
                else:
                    read_image = cv2.imread(image_path)
            else:
                read_image = image

            self.image_path = image_path
            self.image_name = image_name
            self.image_tag = image_type
            self.image = read_image
        else:
            raise Exception("Unable to load image - " + image_path)

    def get_file_with_tag(self):
        return self.image_name + "." + self.image_tag


def clear_screen():
    print('\n'*__CLEAR_SCREEN_COUNT__)


def cv_wait(isLive=False):
    if isLive:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
    else:
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        return True


def breakDownMessage(size=0, msg=""):
    message = []
    current_pos = 0

    for pos in range(0, len(msg)):
        pos = current_pos

        try:
            if msg[pos + size]:
                for space in range(size, 0, -1):
                    if msg[pos + space] == ' ':
                        message.append(msg[pos:pos + space])
                        current_pos = pos + space + 1
                        break
        except IndexError:
            message.append(msg[pos:])
            break
    return message


def fillMsg(msg="", is_title=False):
    if is_title:
        if len(msg) > __MAX_TITLE_SIZE__:
            message = breakDownMessage(__MAX_TITLE_SIZE__, msg)
            for string in message:
                print("| " + string.ljust(__MAX_TITLE_SIZE__) + " |")
        else:
            print("| " + msg.ljust(__MAX_TITLE_SIZE__) + " |")
    else:
        if len(msg) > __MAX_OPT_SIZE__:
            message = breakDownMessage(__MAX_OPT_SIZE__, msg)
            for string in message:
                print("|".ljust(__MAX_BORDER_SIZE__-__MAX_OPT_SIZE__) + string.ljust(__MAX_OPT_SIZE__) + " |")
        else:
            print("|".ljust(__MAX_BORDER_SIZE__-__MAX_OPT_SIZE__) + msg.ljust(__MAX_OPT_SIZE__) + " |")


def fillBorder(border_type=""):
    if border_type == __FILLER__:
        border = "|" + " " * __MAX_BORDER_SIZE__ + "|"
        print(border)
    elif border_type == __HEADER_BORDER__:
        border = "|" + "=" * __MAX_BORDER_SIZE__ + "|"
        print(border)
    elif border_type == __SUB_BORDER__:
        border = "|" + "-" * __MAX_BORDER_SIZE__ + "|"
        print(border)
    else:
        return False


def fillMenu(option=0, msg=""):
    if len(msg) > __MAX_OPT_SIZE__:
        return False
    else:
        print("| " + str(option) + " - " + msg.ljust(__MAX_OPT_SIZE__) + "|")


def getImages(gray=False):
    try:
        while True:
            images = []

            fillBorder(__FILLER__)
            fillBorder(__SUB_BORDER__)
            fillBorder(__FILLER__)
            fillMenu(0, "Go Back")
            fillMenu(1, "Use Default Images for Detection")
            fillMenu(2, "Provide Path to Image for Detection")
            fillBorder(__FILLER__)
            menu_option = input(__INPUT_STR__)

            if menu_option == __OPTION_ONE__:
                files = os.listdir(__IMAGE_GOLDEN_PATH__)
                files.sort()

                for file in files:

                    image = SavedImage(__IMAGE_GOLDEN_PATH__ + file, gray)

                    cv2.imshow("Original - " + image.get_file_with_tag(), image.image)

                    cv_wait()

                    images.append(image)

                return images
            elif menu_option == __OPTION_TWO__:
                fillBorder(__FILLER__)
                fillBorder(__FILLER__)

                path_to_image = input("|  Please provide the path to the image: ")

                image = SavedImage(path_to_image, gray)

                cv2.imshow("Original - " + image.get_file_with_tag(), image.image)

                cv_wait()

                images.append(image)

                return images
            elif menu_option == __OPTION_RETURN__:
                return
            else:
                print("\n\nAn incorrect option has been made! Try again...")
    except Exception as e:
        fillMsg("ERROR! Error message is: " + str(e))


def launchFaceImageClassifier(classifier=None):
    images = getImages()
    if images is not None:
        for image in images:
            faces = classifier.detectMultiScale(image.image, scaleFactor=1.1,
                                                minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(image.image, (x, y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

            cv2.imshow("Viola and Jones Algorithm + Face Detection - " + image.get_file_with_tag(), image.image)

            cv_wait()

            cv2.imwrite(__VIOLA_JONES_RESULTS_PATH__ + image.get_file_with_tag(), image.image)


def launchFaceLiveClassifier(classifier=None):
    live_camera_feed = cv2.VideoCapture(0)

    while True:
        ret, frame = live_camera_feed.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_faces = classifier.detectMultiScale(rgb_frame, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in detected_faces:
            if w > 250:
                cv2.rectangle(frame, (x, y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

        cv2.imshow('Live Feed', frame)

        if cv_wait(isLive=True):
            break;

    # After the loop release the cap object
    live_camera_feed.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def camera_or_image():
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
    # Using the OpenCV prebuilt/taught Cascade Classifiers, store for later use
    cascade_face = cv2.CascadeClassifier(__PROJECT_ROOT__ + __CASCADE_FACE_CLASSIFIER_PATH__)

    while True:
        menu_option = camera_or_image()

        if menu_option == __OPTION_ONE__:
            launchFaceImageClassifier(cascade_face)
        elif menu_option == __OPTION_TWO__:
            launchFaceLiveClassifier(cascade_face)
        elif menu_option == __OPTION_RETURN__:
            break
        else:
            print("\n\nAn incorrect option has been made! Try again...")


def open_cv_window_wait(images=None, title="", saveFile=False, saveFilePath=""):
    for image in images:
        cv2.imshow(title + image.get_file_with_tag(), image.image)

        if saveFile:
            cv2.imwrite(saveFilePath + image.get_file_with_tag(), image.image)

        cv_wait()


def launchFaceImageHOG():
    images = getImages()

    if images is not None:
        image_gradients = []
        for image in images:
            # Convert image to readable matrix to calculate the gradient
            # using Sobel Edge Detection
            img = np.float32(image.image)/255.0

            # Chose a K size of 3 to be able to get the best gradient from any image
            gradX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gradY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

            magnitude, angle = cv2.cartToPolar(gradX, gradY, angleInDegrees=True)

            image_gradients.append(SavedImage("/" + image.get_file_with_tag(), image=magnitude))

        open_cv_window_wait(image_gradients, "Gradients of - ", saveFile=False)

        fillBorder(__FILLER__)
        fillBorder(__FILLER__)
        fillMsg("Those gradients will now be used to calculate the histogram of the images, " +
                  "and unilaterally provide face detection using the DLib library.")

        fillBorder(__FILLER__)
        time.sleep(3)

        # Conduct Face Detection using the above gradient method and DLib Histogram Library
        dlib_face_detection = dlib.get_frontal_face_detector()

        for image in images:
            detected_faces = dlib_face_detection(image.image, 1)

            for (i, rect) in enumerate(detected_faces):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image.image, (x,y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

        open_cv_window_wait(images, "Histogram of Oriented Gradients + Face Detection - ",
                            saveFile=True, saveFilePath=__HOG_RESULTS_PATH__)

        fillBorder(__FILLER__)
        fillBorder(__FILLER__)


def launchFaceLiveHOG():
    live_camera_feed = cv2.VideoCapture(0)
    dlib_face_detection = dlib.get_frontal_face_detector()

    while True:
        ret, frame = live_camera_feed.read()
        live_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_faces = dlib_face_detection(live_frame)

        for (i, rect) in enumerate(detected_faces):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

        cv2.imshow("Live Feed", frame)

        if cv_wait(isLive=True):
            break;

    # After the loop release the cap object
    live_camera_feed.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def launchHistogramOrientedGradients():
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
    images = getImages()

    if images is not None:
        dlib_face_detection = dlib.cnn_face_detection_model_v1(__MODELS_FACE_DLIB_PATH__)

        for image in images:
            detected_faces = dlib_face_detection(image.image)

            for (i, rect) in enumerate(detected_faces):
                x1_left = rect.rect.left()
                y1_top = rect.rect.top()
                x2_right = rect.rect.right()
                y2_bottom = rect.rect.bottom()

                # Draw rectangle around the found face
                cv2.rectangle(image.image, (x1_left, y1_top), (x2_right, y2_bottom), __GREEN_BORDER__, __BORDER_THICKNESS__)

        open_cv_window_wait(images, "Convolutional Neural Networks + Face Detection - ",
                            saveFile=True, saveFilePath=__CNN_RESULTS_PATH__)


def launchFaceLiveCNN():
    live_camera_feed = cv2.VideoCapture(0)
    dlib_face_detection = dlib.cnn_face_detection_model_v1(__MODELS_FACE_DLIB_PATH__)

    while True:
        ret, frame = live_camera_feed.read()
        live_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_faces = dlib_face_detection(live_frame)

        for (i, rect) in enumerate(detected_faces):
            x1_left = rect.rect.left()
            y1_top = rect.rect.top()
            x2_right = rect.rect.right()
            y2_bottom = rect.rect.bottom()

            # Draw rectangle around the found face
            cv2.rectangle(frame, (x1_left, y1_top), (x2_right, y2_bottom), __GREEN_BORDER__,
                          __BORDER_THICKNESS__)

        cv2.imshow("Live Feed", frame)

        if cv_wait(isLive=True):
            break;

    # After the loop release the cap object
    live_camera_feed.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def launchConvolutionalNeuralNetwork():
    while True:
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
    while True:
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

    __PROJECT_ROOT__ = sys.argv[1] + "/"

    start()
