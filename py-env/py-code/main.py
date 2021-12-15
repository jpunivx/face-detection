#!/../bin/python3.9

# Python Imports
import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

# Intro Message
__INTRO_TITLE__ = "Face Detection --- researched and implemented by Joseph Pildush"
__INTRO_MSG__ = "The options below will run different face detection algorithms/techniques. The purpose of this program is " +\
                "to demonstrate the different accuracies and efficiencies of using the below Facial Detection algorithms/techniques " + \
                "on an Image, or during a Live camera feed."

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

# Image
__IMAGE_PERSON_ONE__ = "./images/person1.jpg"
__IMAGE_PERSON_TWO__ = "./images/person2.jpg"
__GREEN_BORDER__ = (0, 255, 0)
__BORDER_THICKNESS__ = 10

# Cascade Paths
__CASCADE_CLASSIFIER_PATH__ = "../lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml"
__CASCADE_EYE_PATH__ = "../lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml"
__CASCADE_SMILE_PATH__ = "../lib/python3.9/site-packages/cv2/data/haarcascade_smile.xml"


def clear_screen():
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')


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


def fillIntro(msg="", is_title=False):
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


def getImages():
    while True:
        images = []
        clear_screen()
        fillBorder(__FILLER__)
        fillBorder(__SUB_BORDER__)
        fillMenu(0, "Go Back")
        fillMenu(1, "Use Default Images for Detection")
        fillMenu(2, "Provide Path to Image for Detection")
        fillBorder(__FILLER__)
        menu_option = input(__INPUT_STR__)

        if menu_option == __OPTION_ONE__:
            image = cv2.imread(__IMAGE_PERSON_ONE__)
            color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.figure().suptitle("Original", fontsize=20)
            plt.imshow(color_img)
            plt.show()

            images.append(color_img)

            image = cv2.imread(__IMAGE_PERSON_TWO__)
            color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.figure().suptitle("Original", fontsize=20)
            plt.imshow(color_img)
            plt.show()

            images.append(color_img)

            return images
        elif menu_option == __OPTION_TWO__:
            fillBorder(__FILLER__)
            fillBorder(__FILLER__)

            path_to_image = input("|  Please provide the path to the image: ")
            image = cv2.imread(path_to_image)
            color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.figure().suptitle("Original", fontsize=20)
            plt.imshow(color_img)
            plt.show()

            images.append(color_img)

            return images
        elif menu_option == __OPTION_RETURN__:
            return
        else:
            print("\n\nAn incorrect option has been made! Try again...")


def launchFaceImageClassifier(classifier=None):
    images = getImages()
    if images is not None:
        for img in images:
            faces = classifier.detectMultiScale(img, scaleFactor=1.1,
                                                minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), __GREEN_BORDER__, __BORDER_THICKNESS__)

            plt.figure().suptitle("Face Detected", fontsize=20)
            plt.imshow(img)
            plt.show()


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
                roi_rgb_frame = rgb_frame[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

        cv2.imshow('Live Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    live_camera_feed.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def launchViolaJonesClassifier():
    # Using the OpenCV prebuilt/taught Cascade Classifiers, store for later use
    cascade_face = cv2.CascadeClassifier(__CASCADE_CLASSIFIER_PATH__)
    cascade_eye = cv2.CascadeClassifier(__CASCADE_EYE_PATH__)
    cascade_smile = cv2.CascadeClassifier(__CASCADE_SMILE_PATH__)

    while True:
        # Request for image to detect face
        clear_screen()
        fillBorder(__SUB_BORDER__)
        fillMenu(0, "Go Back")
        fillMenu(1, "Detect Face in Image")
        fillMenu(2, "Detect Face in Camera")
        fillBorder(__FILLER__)
        menu_option = input(__INPUT_STR__)

        if menu_option == __OPTION_ONE__:
            launchFaceImageClassifier(cascade_face)
        elif menu_option == __OPTION_TWO__:
            launchFaceLiveClassifier(cascade_face)
        elif menu_option == __OPTION_RETURN__:
            break
        else:
            print("\n\nAn incorrect option has been made! Try again...")


def start():
    while True:
        clear_screen()
        fillBorder(__HEADER_BORDER__)
        fillIntro(__INTRO_TITLE__, is_title=True)
        fillBorder(__HEADER_BORDER__)
        fillIntro(__INTRO_MSG__)
        fillBorder(__SUB_BORDER__)
        fillBorder(__FILLER__)
        fillMenu(0, "Exit")
        fillMenu(1, "Viola–Jones Object Detection")
        fillMenu(2, "Histogram of the Oriented Gradients")
        fillMenu(3, "Convolutional Neural Networks (CNN)")
        fillBorder(__FILLER__)

        menu_option = input(__INPUT_STR__)

        if menu_option == __OPTION_ONE__:
            launchViolaJonesClassifier()
            pass
        elif menu_option == __OPTION_TWO__:
            pass
        elif menu_option == __OPTION_THREE__:
            pass
        elif menu_option == __OPTION_RETURN__:
            break
        else:
            print("Invalid Option! Please Try Again!")


if __name__ == '__main__':
    # Launch Options Menu
    start()
