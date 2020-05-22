import cv2
import numpy as np
import matplotlib.pyplot as plt
DEBUG = True

def find_boundary(mean_arr, coef):
    th = np.mean(mean_arr) * coef

    sections = []
    section = []
    for i, val in enumerate(mean_arr):
        if val > th:
            section.append(i)
        else:
            if len(section) != 0:
                sections.append(section)
                section = []

    max_len = 0
    max_section = []
    for section in sections:
        if len(section) > max_len:
            max_section = section
            max_len = len(section)
    return max_section

def projection_scan(image):
    mean_hor = np.mean(image, axis=1)
    mean_ver = np.mean(image, axis=0)
    hor_coef = 1
    ver_coef = 0.8

    max_len_hor = find_boundary(mean_hor, hor_coef)
    max_len_ver = find_boundary(mean_ver, ver_coef)

    start1, end1 = max_len_hor[0], max_len_hor[len(max_len_hor) - 1]
    start2, end2 = max_len_ver[0], max_len_ver[len(max_len_ver) - 1]

    fig = plt.figure(figsize=(12, 10))
    plt.subplot(121)
    plt.plot(mean_hor, 'b-')
    plt.grid(True)
    plt.xlabel("Y axis pixel No.")
    plt.ylabel("Vertical projection value")
    plt.axhline(y=np.mean(mean_hor) * hor_coef, color='k', linestyle='--', linewidth=1)
    plt.title("Vertical Projection")
    plt.subplot(122)
    plt.xlabel("X axis pixel No.")
    plt.ylabel("Horizontal projection value")
    plt.plot(mean_ver, 'g-')
    plt.grid(True)
    plt.title("Horizontal Projection")
    plt.axhline(y=np.mean(mean_ver) * ver_coef, color='k', linestyle='--', linewidth=1)
    plt.show()

    return start1, end1, start2, end2

def detect_nose_band_backup(img, img_hsv, points, hue_thresh):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    cv2.imshow("h", img_hsv)

    lower_green = np.array([hue_thresh-5, 0, 0])  # green
    upper_green = np.array([hue_thresh+70, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    img_green = cv2.bitwise_and(img, img, mask=green_mask)
    cv2.imshow("h", img_green)

    img_h, img_s, img_v = cv2.split(img_green)
    cv2.imshow("img_h", img_h)

    img_canvas = np.zeros((img.shape[0], img.shape[1], 1), np.uint8) * 255 #---black in RGB
    mask_raw1 = cv2.rectangle(img_canvas, p1, p4, (255, 255, 255), -1)  # ---the dimension of the ROI
    ret, mask1 = cv2.threshold(img_canvas, 127, 255, 0)
    img_v = cv2.bitwise_and(img_h, img_h, mask=mask1)
    cv2.imshow("img_v_mask", img_v)
    cv2.waitKey(0)

    top = p2[1]
    th = 127
    mask_len = p4[0] - p2[0]
    nose_piece_th = 0.5 * mask_len
    coord_noseband = None

    zero_section = []
    for i in range(top-20, top-50, -1):
        line_section = img_v[i:i+1, p2[0]:p4[0]].flatten()
        for j, val in enumerate(line_section):
            if val == 0:
                zero_section.append(j)
            else:
                if len(zero_section) > nose_piece_th:
                    width_start = zero_section[0] + p2[0]
                    width_end =  zero_section[len(zero_section)-1] + p2[0]
                    height = i
                    coord_noseband = [width_start, width_end, height]
                    return coord_noseband
                else:
                    zero_section = []
    return coord_noseband

def detect_nose_band(img, img_hsv, points, s_thresh):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    cv2.imshow("hsv", img_hsv)

    h, s, v = cv2.split(img_hsv)
    cv2.imshow("h", h)
    cv2.imshow("s", s)
    cv2.imshow("v", v)
    cv2.waitKey(0)

    img_h, img_s, img_v = cv2.split(img_hsv)
    cv2.imshow("img_s", img_s)

    img_canvas = np.zeros((img.shape[0], img.shape[1], 1), np.uint8) * 255 #---black in RGB
    mask_raw1 = cv2.rectangle(img_canvas, p1, p4, (255, 255, 255), -1)  # ---the dimension of the ROI
    ret, mask1 = cv2.threshold(img_canvas, 127, 255, 0)
    img_v = cv2.bitwise_and(img_h, img_h, mask=mask1)
    cv2.imshow("img_v_mask", img_v)
    cv2.waitKey(0)

    top = p2[1]
    th = 127
    mask_len = p4[0] - p2[0]
    nose_piece_th = 0.5 * mask_len
    coord_noseband = None

    zero_section = []
    for i in range(top-20, top-50, -1):
        line_section = img_v[i:i+1, p2[0]:p4[0]].flatten()
        for j, val in enumerate(line_section):
            if val == 0:
                zero_section.append(j)
            else:
                if len(zero_section) > nose_piece_th:
                    width_start = zero_section[0] + p2[0]
                    width_end =  zero_section[len(zero_section)-1] + p2[0]
                    height = i
                    coord_noseband = [width_start, width_end, height]
                    return coord_noseband
                else:
                    zero_section = []
    return coord_noseband


def mask_inspection(frame, img_hsv, points, s_thresh):
    eval = 'OK'
    # results = []
    # coord_noseband = detect_nose_band(frame, img_hsv, points, s_thresh)
    # # coord_earband = detect_ear_band(frame, img_hsv, points, hue_thresh)
    return  eval


def main():
    cam = cv2.VideoCapture(1)
    if (cam.isOpened() == False):
        print("Error opening video file")
        return False

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("The frame width :{}, the height :{}".format(width, height))

    set_ROI = True
    frame_count = 0
    mask_count = 0
    sample_val_list = []
    mask_gone = False
    mask_new = False
    frame_check_amount = 1
    frame_check = 0
    eval = 'OK'

    out = cv2.VideoWriter('./video_record5.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # hsv image
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv", img_hsv)
        # h, s, v = cv2.split(img_hsv)
        # cv2.imshow("h", h)
        # cv2.imshow("s", s)
        # cv2.imshow("v", v)
        # cv2.waitKey(0)
        # gray image
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        if DEBUG == True:
            # Select ROI
            r1, r2, c1, c2 = cv2.selectROI(frame)
            p1 = (c1, r1)
            p2 = (c1, r2)
            p3 = (c2, r1)
            p4 = (c2, r2)
            points = [p1, p2, p3, p4]
            s_thresh = img_hsv[int((r1 + r2) / 2), int((c1 + c2) / 2)][1]
            coord_noseband = detect_nose_band(frame, img_hsv, points, s_thresh)
            # coord_earband = detect_ear_band(frame, img_hsv, points, hue_thresh)
            return 0

        # set ROI, if not qualified, check next frame
        if set_ROI == True:
            r1, r2, c1, c2 = projection_scan(img_gray)
            ratio = (c1-c2) / (r1-r2)
            print("ratio is:{}".format(ratio))
            # get sample area for counting mask
            center_r = int((r1 + r2) / 2)
            center_c = int((c1 + c2) / 2)
            if ratio < 2 and ratio > 1.5:
                set_ROI = False
            else:
                continue

        # mask tracking
        if mask_gone == False:
            sample_val = np.mean(img_gray[center_c - 10: center_c + 10, center_r - 10: center_c + 10])
            sample_val_list.append(sample_val)
            sample_val_mean = np.mean(sample_val_list)
            # print(sample_val, sample_val_mean)
            if sample_val < sample_val_mean * 0.7:
                print("warning......droping......")
                mask_gone = True
                mask_new = False

        else:
            para = 10
            sample_val = np.mean(img_gray[center_c - 50: center_c + 50, center_r - 5: center_c + 5])
            if sample_val > sample_val_mean * 0.9:
                print("elevating!!!!!!!!!!!!!!")
                mask_new = True
                mask_gone = False
                mask_count += 1

        if mask_new == True:
            # Region of interest
            rect_p1 = (c1, r1)
            rect_p2 = (c2, r2)
            # general four corners
            p1 = (c1, r1)
            p2 = (c1, r2)
            p3 = (c2, r1)
            p4 = (c2, r2)
            points = [p1, p2, p3, p4]
            hue_thresh = img_hsv[int((r1 + r2) / 2), int((c1 + c2) / 2)][0]
            # print("the current hue thresh :{}".format(hue_thresh))

            if frame_check < frame_check_amount:
                eval = mask_inspection(frame, img_hsv, points, hue_thresh)
                frame_check += 1

            cv2.rectangle(frame, rect_p1, rect_p2, (0, 255, 0), 3)
            cv2.putText(frame, "Mask No: {}".format(mask_count), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, "Inspection: {}".format(eval), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(frame, "Frame No: {}".format(frame_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Mask AOI", frame)

        frame_count += 1
        # Saves for video
        out.write(frame)
        # if frame_count < 700:
        #     out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
