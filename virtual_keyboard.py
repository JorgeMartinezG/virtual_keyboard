import cv2
import numpy as np
import string


def get_contour_params(contour):
    (x_pos, y_pos), radius = cv2.minEnclosingCircle(contour)

    return {
        "center": (int(x_pos), int(y_pos)),
        "radius": radius,
        "area": cv2.contourArea(contour)
    }


def draw_rectangle(img_frame, coords):
    """
    Renders rectangles given coordinates and image frame.
    Input
    =====
    img_frame: numpy array
    coords: Tuple with x1, y1, x2, y2 coordinates
    """
    x_point, y_point, width, height = coords
    cv2.rectangle(
        img_frame,
        (x_point, y_point),
        (x_point + width, y_point + height),
        (0, 255, 0),
        2
    )


def binarize_image(img, threshold_value):
    # Transform into grayscale image.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur image using a mean filter.
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Threshold image.
    _, img_thresh = cv2.threshold(img_blur,
                                  threshold_value,
                                  255,
                                  cv2.THRESH_BINARY_INV)

    # perform image dilation with squared kernel
    kernel = np.ones((5, 5), np.uint8)
    img_thresh = cv2.dilate(img_thresh, kernel, iterations=2)

    return img_thresh


def create_opencv_capture():
    capture = cv2.VideoCapture(0)

    # Set frame resolution fixed. 
    width, height = 640, 480
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, height)    

    return capture


def draw_letters(frame):
    x_pos, y_pos = 0, 70
    #define font and text color
    font_params = {
        "type": cv2.FONT_HERSHEY_SIMPLEX,
        "color": (0, 0, 255),
        "size": 2,
        "width": 3
    }

    # Put letters into the image.
    for letter in list(string.ascii_lowercase):
        x_pos += 50
        cv2.putText(frame,
                    letter,
                    (x_pos, y_pos),
                    font_params["type"],
                    font_params["size"],
                    font_params["color"],
                    font_params["width"])


def draw_crosshair(img, params):
    color = (0, 0, 0)
    cv2.circle(img,
               params["center"],
               int(params["radius"]),
               color,
               thickness=5)
    cv2.line(img,
             (params["center"][0] - 50, params["center"][1]),
             (params["center"][0] + 50, params["center"][1]),
             color,
             3)
    cv2.line(img,
             (params["center"][0], params["center"][1] - 50),
             (params["center"][0], params["center"][1] + 50),
             color,
             3)


def create_trackbar(named_window):
    def nothing(x):
        pass

    cv2.namedWindow(named_window)
    cv2.createTrackbar("threshold", named_window, 0, 255, nothing)

    # Set starting value for thresholding.
    cv2.setTrackbarPos("threshold", named_window, 65)


def get_image(cap):
    flag, img = cap.read()
    img = cv2.flip(img, 1)
    composite = cv2.resize(img.copy(), (800, 600))

    return composite


def main():
    named_window = "Image"
    # Create Video capture from opencv.
    cap = cv2.VideoCapture(1)

    # Create Image window and trackbar.
    create_trackbar(named_window)

    while cap.isOpened():
        composite = get_image(cap)

        # Work only within the keyboard section.
        keyboard_section = composite.copy()[20:100, :composite.shape[1]]

        # Drawing letters in the captured frame.
        draw_letters(composite)

        # Transform into a binary image.
        threshold_value = cv2.getTrackbarPos("threshold", "Image")
        img_thresh = binarize_image(keyboard_section, threshold_value)

        # Find contours in the thresholded image.
        contours, hierarchy = cv2.findContours(img_thresh.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_params = get_contour_params(contour)
            # Draw crosshair.
            if contour_params["area"] > 700 and contour_params["area"] < 2000:
                draw_crosshair(composite, contour_params)

        img_stacked = np.vstack([img_thresh, cv2.cvtColor(composite,
                                                          cv2.COLOR_BGR2GRAY)])

        cv2.imshow(named_window, img_stacked)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
    print("Done")


if __name__ == '__main__':
    main()    