import cv2
import numpy as np
import matplotlib.pyplot as plt

def write_log():
    pass


def frame_process(img, counter):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    cv2.imshow("test", h)

    th, img_threshold = cv2.threshold(h, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow("h", img_threshold)
    kernel = np.ones((15, 15), np.uint8)
    img_opening = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening", img_opening)

    contours, hierarchy = cv2.findContours(img_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    new_contours=contours[max_index]

    cv2.drawContours(img, new_contours, -1, (255, 0, 0), -1)
    img_black = np.zeros((img_opening.shape[0], img_opening.shape[1], 1), np.uint8)
    cv2.drawContours(img_black, new_contours, -1, 255, -1)  # contourIdx –1, means all the contours are drawn.
    cv2.imshow("canvas1", img_black)

    section = img_black[10:11, 0:640].flatten().tolist()
    if 255 in section:
        section_col = section.index(255)
    else:
        line_points = None

    section1 = img_black[200:201, 0:640].flatten().tolist()
    if 255 in section1:
        section1_col = section1.index(255)
    else:
        line_points = None
    distance = (section1_col + section_col)/2
    p1 = (section_col, 10)
    p2 = (section1_col, 200)
    line_points = [p1, p2, distance]
    return line_points

def main():
    cam = cv2.VideoCapture(0)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("The frame width :{}, the height :{}".format(width, height))
    counter = 1
    out = cv2.VideoWriter('./out_test_s2.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (width, height))

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        img = frame.copy()
        line_points = frame_process(img, counter)
        if line_points == None:
            print("someting worong")
        else:
            cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 3)
            cv2.putText(frame, "Distance with baseline: {}".format(line_points[2]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, "Frame No: {}".format(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Mask AOI", frame)

        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if counter < 600:
            out.write(frame)
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
