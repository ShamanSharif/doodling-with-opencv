import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale=0.75):
    """Works on Images, Videos and Live Videos"""
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimension = (width, height)

    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


def changeResolution(capture, width, height):
    """Works only on Live Videos"""
    capture.set(3, width)
    capture.set(4, height)
    return capture


def showImage():
    img = cv.imread("Photos/cat.jpg")
    image_resized = rescaleFrame(img)
    cv.imshow("Cat", image_resized)
    cv.waitKey(0)


def showVideo():
    # capture = cv.VideoCapture(0)
    capture = cv.VideoCapture("Videos/dog.mp4")
    while True:
        isTrue, frame = capture.read()
        # frame_resized = rescaleFrame(frame)
        # cv.imshow("Video", frame_resized)
        cv.imshow("Video", frame)

        if cv.waitKey(20) & 0xFF == ord("d"):
            break

    capture.release()
    cv.destroyAllWindows()


def showLiveVideo():
    capture = cv.VideoCapture(0)
    capture = changeResolution(capture, 250, 250)
    while True:
        isTrue, frame = capture.read()
        # frame_resized = rescaleFrame(frame)
        # cv.imshow("Video", frame_resized)
        cv.imshow("Video", frame)

        if cv.waitKey(20) & 0xFF == ord("d"):
            break

    capture.release()
    cv.destroyAllWindows()


def drawCV():
    blank = np.zeros((500, 500, 3), dtype="uint8")
    cv.imshow("Blank", blank)

    cv.rectangle(
        blank,
        (0, 0),
        (blank.shape[1] // 2, blank.shape[0] // 2),
        (0, 255, 0),
        thickness=cv.FILLED,
    )
    cv.imshow("Blank2", blank)

    cv.circle(
        blank,
        (blank.shape[1] // 2, blank.shape[0] // 2),
        blank.shape[1] // 4,
        (0, 0, 255),
        thickness=cv.FILLED,
    )
    cv.imshow("Blank3", blank)

    cv.line(
        blank,
        (0, 0),
        (blank.shape[1] // 2, blank.shape[0] // 2),
        (0, 0, 0),
        thickness=2,
    )
    cv.imshow("Blank4", blank)

    cv.putText(
        blank,
        "Coffee",
        (blank.shape[1] // 2, blank.shape[0] // 2),
        cv.FONT_HERSHEY_DUPLEX,
        1.0,
        (0, 0, 0),
        thickness=2,
    )
    cv.imshow("Blank5", blank)

    cv.waitKey(0)


def main():
    # showImage()
    # showVideo()
    # showLiveVideo()
    drawCV()
    print("Hello OpenCV")


if __name__ == "__main__":
    main()
