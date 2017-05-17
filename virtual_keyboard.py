import cv2
import numpy as np


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


def main():
    # Create Video capture from opencv.
    cap = cv2.VideoCapture(0)

    # for i in range(16):
    #     flag, img = cap.read()
    #     if not flag:
    #         break
    #     out = fg_bg.apply(img, learningRate=0.5)

    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            break

        # hands = classifier.detectMultiScale(
        #     img,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     minSize=(30, 30),
        #     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        # )


        # for hand in hands:
        #     draw_rectangle(img, hand)

        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
        ret, img_thresh = cv2.threshold(img_blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        nonzero_pixels = cv2.findNonZero(img_thresh)
        import ipdb; ipdb.set_trace()

        # for contour in contours:
        #     (x,y),radius = cv2.minEnclosingCircle(contour)
        #     center = (int(x),int(y))
        #     area = cv2.contourArea(contour)
        #     if area > 1000 and area < 2000:
        #         cv2.circle(img, center, int(radius),(0,120,255), thickness=5)
        #         cv2.line(img, (center[0]-50,center[1]), (center[0]+50,center[1]), (0,0,0),3)
        #         cv2.line(img, (center[0],center[1]-50), (center[0],center[1]+50), (0,0,0),3)

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
        # cv2.imshow('Keyboard', img)
        # key = cv2.waitKey(25) & 0xFF
        # if key == ord('q'):
        #     break


if __name__ == '__main__':
    main()    