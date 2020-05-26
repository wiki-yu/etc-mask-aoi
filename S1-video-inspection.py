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

    # fig = plt.figure(figsize=(12, 10))
    # plt.subplot(121)
    # plt.plot(mean_hor, 'b-')
    # plt.grid(True)
    # plt.xlabel("Y axis pixel No.")
    # plt.ylabel("Vertical projection value")
    # plt.axhline(y=np.mean(mean_hor) * hor_coef, color='k', linestyle='--', linewidth=1)
    # plt.title("Vertical Projection")
    # plt.subplot(122)
    # plt.xlabel("X axis pixel No.")
    # plt.ylabel("Horizontal projection value")
    # plt.plot(mean_ver, 'g-')
    # plt.grid(True)
    # plt.title("Horizontal Projection")
    # plt.axhline(y=np.mean(mean_ver) * ver_coef, color='k', linestyle='--', linewidth=1)
    # plt.show()

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

def check_left_earband(img_processed, rect_p1, rect_p2):
    for i in range(rect_p1[0], rect_p1[0] + 30):
        line_section = img_processed[rect_p1[1]:rect_p2[1], i:i+1].flatten()
        # plt.figure(figsize=(12, 5))
        # plt.plot(line_section, 'b-')
        # # plt.axhline(y=175, color='r', linestyle='--', linewidth=1)
        # plt.grid(True)
        # plt.show()

        sections = []
        section = []
        for j, val in enumerate(line_section):
            if val > 250:
                section.append(j)
            else:
                if len(section) != 0:
                    sections.append(section)
                    section = []

        if len(sections) == 2:
            if 5 < len(sections[0]) and len(sections[0]) < 25 and 5 < len(sections[1]) and len(sections[1]) < 25:
                earband_pos = [i, sections[0][0]+rect_p1[1], sections[1][0]+rect_p1[1]]
                print("left: {}".format(earband_pos))
                return earband_pos

    return None

def check_right_earband(img_processed, rect_p3, rect_p4):
    for i in range(rect_p3[0], rect_p3[0]-30, -1):
        line_section = img_processed[rect_p3[1]:rect_p4[1], i:i + 1].flatten()
        # plt.figure(figsize=(12, 5))
        # plt.plot(line_section, 'b-')
        # # plt.axhline(y=175, color='r', linestyle='--', linewidth=1)
        # plt.grid(True)
        # plt.show()

        sections = []
        section = []
        for j, val in enumerate(line_section):
            if val > 250:
                section.append(j)
            else:
                if len(section) != 0:
                    sections.append(section)
                    section = []

        if len(sections) == 2:
            if 5 < len(sections[0]) and len(sections[0]) < 25 and 5 < len(sections[1]) and len(sections[1]) < 25:
                earband_pos = [i, sections[0][0] + rect_p3[1], sections[1][0] + rect_p3[1]]
                print("right: {}".format(earband_pos))
                return earband_pos

    return None

def check_right_earband(img_processed, rect_p3, rect_p4):
    for i in range(rect_p3[0], rect_p3[0]-30, -1):
        line_section = img_processed[rect_p3[1]:rect_p4[1], i:i + 1].flatten()
        # plt.figure(figsize=(12, 5))
        # plt.plot(line_section, 'b-')
        # # plt.axhline(y=175, color='r', linestyle='--', linewidth=1)
        # plt.grid(True)
        # plt.show()

        sections = []
        section = []
        for j, val in enumerate(line_section):
            if val > 250:
                section.append(j)
            else:
                if len(section) != 0:
                    sections.append(section)
                    section = []

        if len(sections) == 2:
            if 5 < len(sections[0]) and len(sections[0]) < 25 and 5 < len(sections[1]) and len(sections[1]) < 25:
                earband_pos = [i, sections[0][0] + rect_p3[1], sections[1][0] + rect_p3[1]]
                print("right: {}".format(earband_pos))
                return earband_pos

    return None

def check_left_welding(img_processed1, rect_p1, rect_p2):
    for i in range(rect_p1[0], rect_p1[0] + 50):
        line_section = img_processed1[rect_p1[1]:rect_p2[1], i:i+1].flatten()
        # plt.figure(figsize=(12, 5))
        # plt.plot(line_section, 'b-')
        # # plt.axhline(y=175, color='r', linestyle='--', linewidth=1)
        # plt.grid(True)
        # plt.show()

        # sections = []
        # section = []
        # for j, val in enumerate(line_section):
        #     if val > 250:
        #         section.append(j)
        #     else:
        #         if len(section) != 0:
        #             sections.append(section)
        #             section = []
        #
        # if len(sections) == 2:
        #     if 5 < len(sections[0]) and len(sections[0]) < 25 and 5 < len(sections[1]) and len(sections[1]) < 25:
        #         earband_pos = [i, sections[0][0]+rect_p1[1], sections[1][0]+rect_p1[1]]
        #         print("left: {}".format(earband_pos))
        #         return earband_pos

    return None

def detect_ear_band(img, img_hsv, points, mask_count):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    # cv2.imshow("hsv", img_hsv)
    # print("points are :{}".format(points))

    img_h, img_s, img_v = cv2.split(img_hsv)
    # cv2.imshow("img_h", img_h)
    # cv2.imshow("img_s", img_s)
    # cv2.imshow("img_v", img_v)

    reverse_s = cv2.bitwise_not(img_s)
    # cv2.imshow("reverse", reverse_s)

    ret, s_thresh = cv2.threshold(reverse_s, 127, 255, 0)
    # cv2.imshow("thresh", s_thresh)
    # cv2.waitKey(0)

    para = 30
    rect1 = (p1[0]-para, p1[1]-para)
    rect2 = (p4[0]+para, p4[1]+0)
    img_canvas = np.zeros((img.shape[0], img.shape[1], 1), np.uint8) * 255 #---black in RGB
    mask_raw1 = cv2.rectangle(img_canvas, rect1, rect2, (255, 255, 255), -1)  # ---the dimension of the ROI
    ret, mask1 = cv2.threshold(img_canvas, 127, 255, 0)
    img_processed = cv2.bitwise_and(s_thresh, s_thresh, mask=mask1)
    cv2.imwrite("./test/{}.jpg".format(mask_count), img_processed)
    # cv2.imshow("img_processed", img_processed)

    rect_p1 = (p1[0]-para, p1[1]-para)
    rect_p2 = (p2[0]-para, p2[1])
    rect_p3 = (p3[0]+para, p3[1]-para)
    rect_p4 = (p4[0]+para, p4[1])

    earband_pos = []

    left_earbands_pos = check_left_earband(img_processed, rect_p1, rect_p2)
    earband_pos.append(left_earbands_pos)
    # if left_earbands_pos != None:
    #     img_canvas1 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8) * 255  # ---black in RGB
    #     mask_raw1 = cv2.rectangle(img_canvas, rect1, rect2, (255, 255, 255), -1)  # ---the dimension of the ROI
    #     ret, mask1 = cv2.threshold(img_canvas1, 127, 255, 0)
    #     img_processed1 = cv2.bitwise_and(img_s, img_s, mask=mask_raw1)
    #     cv2.imwrite("./test1/{}.jpg".format(mask_count), img_processed1)
    #     check_left_welding(img_processed1, rect_p1, rect_p2)
    #     # cv2.imshow("img_processed1", img_processed1)

    right_earbands_pos = check_right_earband(img_processed, rect_p3, rect_p4)
    earband_pos.append(right_earbands_pos)

    return earband_pos


def mask_inspection(frame, img_hsv, points, mask_count):
    eval = 'OK'
    coord_noseband = detect_ear_band(frame, img_hsv, points, mask_count)

    return coord_noseband, eval

def main():
    cam = cv2.VideoCapture('video_record1.avi')
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

    out = cv2.VideoWriter('./result.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # hsv image
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv", img_hsv)
        h, s, v = cv2.split(img_hsv)
        # cv2.imshow("h", h)
        # cv2.imshow("s", s)
        # cv2.imshow("v", v)
        # cv2.waitKey(0)
        # gray image
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        if DEBUG != True:
            # Select ROI
            val = cv2.selectROI(frame)
            c1 = val[0]
            c2 = val[0] + val[2]
            r1 = val[1]
            r2 = val[1] + val[3]
            p1 = (c1, r1)
            p2 = (c1, r2)
            p3 = (c2, r1)
            p4 = (c2, r2)
            points = [p1, p2, p3, p4]
            s_thresh = img_hsv[int((r1 + r2) / 2), int((c1 + c2) / 2)]
            coord_noseband = detect_ear_band(frame, img_hsv, points, s_thresh[1])
            print(s_thresh)
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
                print("ROI captured at frame No: {}".format(frame_count))
            else:
                continue

        # mask tracking
        if mask_gone == False:
            sample_val = np.mean(h[center_c - 50: center_c + 50, center_r - 5: center_c + 5])
            if sample_val < 90:
                # print("droping, current frame no:{}".format(frame_count))
                mask_gone = True
                mask_new = False
                frame_check = 0
        else:
            sample_val = np.mean(h[center_c - 50: center_c + 50, center_r - 5: center_c + 5])
            if sample_val > 90:
                # print("elevating!!!!!!!!!!!!!!")
                mask_new = True
                mask_gone = False
                mask_count += 1
                print("mask No :{}".format(mask_count))

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
            # hue_thresh = img_hsv[int((r1 + r2) / 2), int((c1 + c2) / 2)][0]
            # print("the current hue thresh :{}".format(hue_thresh))

            if frame_check < frame_check_amount:
                coord, eval = mask_inspection(frame, img_hsv, points, mask_count)
                coord_left = coord[0]
                if coord_left == None:
                    coord_left = [100, 100, 100]
                coord_right = coord[1]
                if coord_right == None:
                    coord_right = [100, 100, 100]
                frame_check += 1

            cv2.rectangle(frame, rect_p1, rect_p2, (0, 255, 0), 3)
            cv2.putText(frame, "Mask No: {}".format(mask_count), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, "Inspection: {}".format(eval), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, "No.1 welding : ({}, {})".format(coord_left[0], coord_left[1]), (coord_left[0], coord_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "No.2 welding : ({}, {})".format(coord_left[0], coord_left[2]), (coord_left[0], coord_left[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "No.3 welding : ({}, {})".format(coord_right[0], coord_right[1]), (coord_right[0], coord_right[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "No.4 welding : ({}, {})".format(coord_right[0], coord_right[2]), (coord_right[0], coord_right[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, "Frame No: {}".format(frame_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Mask AOI", frame)
        out.write(frame)
        cv2.waitKey(10)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
