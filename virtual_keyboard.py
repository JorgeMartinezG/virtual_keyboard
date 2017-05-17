import cv
import cv2
import numpy as np
import string

#define alphabet position
position = 0
#define hand position
hand_position = 0,0
hand_on_keyboard = False
letter_selected = False


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


def binarize_image(img):
    # Transform into grayscale image.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur image using a mean filter.
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Threshold image.
    _, img_thresh = cv2.threshold(img_blur,
                                  70,
                                  255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return img_thresh


def create_opencv_capture():
    capture = cv2.VideoCapture(0)

    # Set frame resolution fixed. 
    width, height = 1920, 1080
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, height)    

    return capture


def draw_letters(frame):
    x_pos, y_pos = 0, 50
    #define font and text color
    font_params = {
        "type": cv2.FONT_HERSHEY_SIMPLEX,
        "color": (0, 0, 255),
        "size": 2,
        "width": 3
    }

    # Put letters into the image.
    for letter in list(string.ascii_lowercase):
        x_pos += 60
        cv2.putText(frame,
                    letter,
                    (x_pos, y_pos),
                    font_params["type"],
                    font_params["size"],
                    font_params["color"],
                    font_params["width"])


def main():
    # Create Video capture from opencv.
    cap = create_opencv_capture()
    fgbg = cv2.BackgroundSubtractorMOG()

    while cap.isOpened():
        flag, img = cap.read()
        composite = img.copy()
        if not flag:
            break
        # Work only within the keyboard section.
        keyboard_section = img[50:200, 0:img.shape[1]]

        # Drawing letters in the captured frame.
        draw_letters(composite)

        # Transform into a binary image.
        # img_thresh = binarize_image(keyboard_section)
        img_thresh = fgbg.apply(keyboard_section)


        # Find contours in the thresholded image.
        contours, hierarchy = cv2.findContours(img_thresh.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            area = cv2.contourArea(contour)
            print area
            if area > 500 and area < 1500:
                cv2.circle(composite, center, int(radius),(0, 0, 0), thickness=5)
                cv2.line(composite, (center[0]-50,center[1]), (center[0]+50,center[1]), (0,0,0),3)
                cv2.line(composite, (center[0],center[1]-50), (center[0],center[1]+50), (0,0,0),3)

        # for cnt in contours:
        #     (x,y),radius = cv2.minEnclosingCircle(cnt)
        #     center = (int(x),int(y))
        #     area = cv2.contourArea(cnt)
        #     if area > 1000 and area < 2000:
        #         radius = int(radius)
        #         cv2.circle(composite,center,radius,(0,120,255), thickness=5)
        #         cv2.line(composite, (center[0]-50,center[1]), (center[0]+50,center[1]), (0,0,0),3)
        #         cv2.line(composite, (center[0],center[1]-50), (center[0],center[1]+50), (0,0,0),3)


        # cv2.drawContours(img_thresh, contours,-1,(0,255,0),3)
        # max_area = 0
        # for i in range(len(contours)):
        #             cnt=contours[i]
        #             area = cv2.contourArea(cnt)
        #             if(area>max_area):
        #                 max_area=area
        #                 ci=i
        # cnt=contours[ci]
        # hull = cv2.convexHull(cnt)
        # drawing = np.zeros(img.shape,np.uint8)
        # cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
        # cv2.drawContours(drawing,[hull],0,(0,0,255),2)
        # out = fg_bg.apply(img_thresh, learningRate=0)

        # Display.
        cv2.imshow('image', composite)
        cv2.imshow('Keyboard', img_thresh)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
    print("Done")

if __name__ == '__main__':
    main()    