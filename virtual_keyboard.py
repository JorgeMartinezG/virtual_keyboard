import collections
import cv2
import math
import numpy as np
import optparse
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


def check_counter(counter):
    if len(counter.most_common()) != 0:
        if counter.most_common(1)[0][1] == 10:
            return True

    return False


def compute_distances(center, letters):
    return [l2_norm(center, letter["position"]) for letter in letters]


def l2_norm(point_1, point_2):
    return math.sqrt((point_2[1] - point_1[1])**2 + 
                     (point_2[0] - point_1[0])**2)


def create_opencv_capture():
    capture = cv2.VideoCapture(0)

    # Set frame resolution fixed. 
    width, height = 640, 480
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, height)    

    return capture


def swipe_letters(contour_params, previous_position, letter_position):
    if previous_position[0] > contour_params["center"][0]:
        letter_position -=100
    elif previous_position[0] < contour_params["center"][0]:
        letter_position += 100

    return letter_position


def draw_letters(frame, x_pos):
    y_pos = 70

    # Put letters into the image.
    letters_position = []
    for letter in list(string.ascii_lowercase):
        x_pos += 50
        cv2.putText(frame,
                    letter,
                    (x_pos, y_pos),
                    font_params["type"],
                    font_params["size"],
                    font_params["color"],
                    font_params["width"])
        letters_position.append({"letter": letter, "position": (x_pos, y_pos)})

    # Add a line to show where the keyboard starts.
    cv2.line(frame, (frame.shape[1], 100), (0, 100), (0, 255, 0), 3)

    return letters_position


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
    cv2.setTrackbarPos("threshold", named_window, options.threshold)


def get_image(cap):
    flag, img = cap.read()
    img = cv2.flip(img, 1)
    composite = cv2.resize(img.copy(), (800, 600))

    return composite


def main():
    named_window = "Image"

    # Create Video capture from opencv.
    cap = cv2.VideoCapture(0)
    cnt_th_low, cnt_th_high = 700, 2000

    # Create Image window and trackbar.
    if options.calibrate:
        create_trackbar(named_window)
    letter_position = 0

    # Create letters counter.
    if not options.calibrate:
        counter = collections.Counter()
        previous_position = (0, 0)
        selected_letters = ""
    threshold_value = options.threshold

    while cap.isOpened():
        composite = get_image(cap)

        # Work only within the keyboard section.
        keyboard_section = composite.copy()[20:100, :composite.shape[1]]

        # Drawing letters in the captured frame.
        if not options.calibrate:
            letters = draw_letters(composite, letter_position)

        # Transform into a binary image.
        if options.calibrate:
            threshold_value = cv2.getTrackbarPos("threshold", "Image")
            cv2.putText(composite,
                        "Threshold value: %d" % threshold_value,
                        (20, composite.shape[0] - 20),
                        font_params["type"],
                        1,
                        (255, 255, 255),
                        2)

        img_thresh = binarize_image(keyboard_section, threshold_value)

        # Find contours in the thresholded image.
        contours, hierarchy = cv2.findContours(img_thresh.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_params = get_contour_params(contour)
            if options.calibrate:
                cv2.putText(composite,
                            "Contour Area: %d" % contour_params["area"],
                            (20, composite.shape[0] - 50),
                            font_params["type"],
                            1,
                            (255, 255, 255),
                            2)

                continue

            # Draw crosshair.
            if (contour_params["area"] > cnt_th_low and
                contour_params["area"] < cnt_th_high):
                draw_crosshair(composite, contour_params)
                distances = compute_distances(contour_params["center"], letters)     

                # Finding letter with closest distance to the crosshair.  
                selected_letter = letters[distances.index(min(distances))]
                counter[selected_letter["letter"]] += 1

            # Swipe.
            if contour_params["area"] > cnt_th_high:
                letter_position = swipe_letters(contour_params,
                                                previous_position,
                                                letter_position)
                previous_position = contour_params["center"]

        if not options.calibrate:
            if check_counter(counter):
                selected_letters += counter.most_common(1)[0][0]
                counter.clear()

            # Displaying letters.
            cv2.putText(composite,
                        selected_letters,
                        (20, 20),
                        font_params["type"],
                        1,
                        (0, 255, 0),
                        2)

            out_img = composite.copy()
        else:
            out_img = np.vstack([img_thresh, cv2.cvtColor(composite,
                                                      cv2.COLOR_BGR2GRAY)])

        # Display output.
        cv2.imshow(named_window, out_img)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            selected_letters = selected_letters[:-1]
        elif key == ord('c'):
            selected_letters = ""

    if options.calibrate:
        print("Final threshold value %d" % threshold_value)
    print("Done")


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-c', '--calibrate',
                      dest='calibrate',
                      action='store_true',
                      default=False)
    parser.add_option('-t', '--threshold',
                      dest='threshold',
                      type='int',
                      default=70)
    options, remainder = parser.parse_args()


    #define font and text color
    font_params = {
        "type": cv2.FONT_HERSHEY_SIMPLEX,
        "color": (0, 255, 0),
        "size": 2,
        "width": 3
    }

    main()