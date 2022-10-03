import cv2
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def detect_lane(frame, height, width):

    frame_0 = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    h_channel = frame_0[:, :, 0]
    cv2.imshow("h_channel image", h_channel)
    # cv2.waitKey(0)
    l_channel = frame_0[:, :, 1]
    cv2.imshow("l_channel image", l_channel)
    # cv2.waitKey(0)
    s_channel = frame_0[:, :, 2]
    cv2.imshow("s_channel image", s_channel)
    cv2.imwrite("s_channel image.jpeg", s_channel)
    # cv2.waitKey(0)

    Z = s_channel.reshape((-1, 1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((s_channel.shape))
    cv2.imshow("threshold after s_channel image", res2)
    cv2.imwrite('threshold after s_channel image.jpeg', res2)
    # cv2.waitKey(0)
    print(center)
    max_num = max(center[0][0], max(center[1][0], center[2][0]))
    min_num = min(center[0][0], min(center[1][0], center[2][0]))
    print(max_num)
    print(min_num)
    for i in range(K):
        if center[i][0] != max_num and center[i][0] != min_num:
            mid_num = center[i][0]
    ret, binary_0 = cv2.threshold(res2, min_num + 1, 255, cv2.THRESH_BINARY)
    print(binary_0)
    cv2.imshow("threshold after s_channel image", binary_0)
    cv2.imwrite('threshold after s_channel image.jpeg', binary_0)
    # cv2.waitKey(0)

    '''frame_0 = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h_channel = frame_0[:, :, 0]
    cv2.imshow("one image", h_channel)
    cv2.waitKey(0)
    s_channel = frame_0[:, :, 1]
    cv2.imshow("one image", s_channel)
    cv2.waitKey(0)
    v_channel = frame_0[:, :, 2]
    cv2.imshow("one image", v_channel)
    cv2.waitKey(0)
    frame_1 = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            if s_channel[i][j] >= 35 and ((h_channel[i][j] >= 0 and h_channel[i][j] < 24) or h_channel[i][j] >= 140):
                frame_1[i][j] = 255
    cv2.imshow("one image", frame_1)
    cv2.waitKey(0)'''

    frame_0 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray image", frame_0)
    cv2.imwrite('gray image.jpeg', frame_0)
    # cv2.waitKey(0)

    edges = cv2.Sobel(frame_0, cv2.CV_16S, 1, 1)
    edges = cv2.convertScaleAbs(edges)
    cv2.imshow("sobel image", edges)
    cv2.imwrite('sobel image.jpeg', edges)
    # cv2.waitKey(0)

    edgesh = cv2.Sobel(frame_0, cv2.CV_16S, 1, 0)
    edgesh = cv2.convertScaleAbs(edgesh)
    cv2.imshow("sobel horizontal image", edgesh)
    cv2.imwrite('sobel horizontal image.jpeg', edgesh)
    # cv2.waitKey(0)

    edgesv = cv2.Sobel(frame_0, cv2.CV_16S, 0, 1)
    edgesv = cv2.convertScaleAbs(edgesv)
    cv2.imshow("sobel vertical image", edgesv)
    cv2.imwrite('sobel vertical image.jpeg', edgesv)
    # cv2.waitKey(0)

    grad = cv2.addWeighted(edgesh, 0.5, edgesv, 0.5, 0)
    cv2.imshow("vertical + horizontal image", grad)
    cv2.imwrite('vertical + horizontal image.jpeg', grad)
    # cv2.waitKey(0)

    gradd = cv2.addWeighted(grad, 0.7, edges, 0.3, 0)
    cv2.imshow("sobel + vertical + horizontal image", gradd)
    cv2.imwrite('sobel + vertical + horizontal image.jpeg', gradd)
    # cv2.waitKey(0)

    graddd = cv2.addWeighted(gradd, 0.75, binary_0, 0.25, 0)
    cv2.imshow("sobel + vertical + horizontal + s_channel image", graddd)
    cv2.imwrite('sobel + vertical + horizontal + s_channel image.jpeg', graddd)
    # cv2.waitKey(0)

    frame_hsv_0 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_hsv = cv2.cvtColor(frame_hsv_0, cv2.COLOR_BGR2HSV)
    lower_yw = np.array([15, 85, 85], dtype="uint8")
    upper_yw = np.array([30, 255, 255], dtype="uint8")
    mask_yw = cv2.inRange(frame_hsv, lower_yw, upper_yw)
    mask_wt = cv2.inRange(frame_0, 200, 255)
    mask = cv2.bitwise_or(mask_wt, mask_yw)
    frame_mask = cv2.bitwise_and(frame_0, mask)
    cv2.imshow("mask image", frame_mask)
    cv2.imwrite('mask image.jpeg', frame_mask)
    # cv2.waitKey(0)

    graddd = cv2.addWeighted(graddd, 0.8, frame_mask, 0.2, 0)
    cv2.imshow("sobel + vertical + horizontal + s_channel + ROI image", graddd)
    cv2.imwrite('sobel + vertical + horizontal + s_channel + ROI image.jpeg', graddd)
    # cv2.waitKey(0)

    Z = graddd.reshape((-1, 1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    print(center)
    res = center[label.flatten()]
    res2 = res.reshape((graddd.shape))
    min_num = min(center[0][0], min(center[1][0], center[2][0]))
    ret, binary = cv2.threshold(res2, min_num + 1, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("threshold after sobel + vertical + horizontal + s_channel image", binary)
    cv2.imwrite('threshold after sobel + vertical + horizontal + s_channel image.jpeg', binary)
    # cv2.waitKey(0)
    print(binary)

    # kernel_size = 9
    kernel_size_temp = round(((height * width) / (80 * 80)) ** 0.5)
    if kernel_size_temp % 2 == 0:
        kernel_size_temp = kernel_size_temp + 1
    kernel_size = max(9, kernel_size_temp)
    print(kernel_size)
    frame_1 = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 1)
    cv2.imshow("GaussianBlur image", frame_1)
    cv2.imwrite('GaussianBlur image.jpeg', frame_1)
    # cv2.waitKey(0)

    # kernel_size = 5
    kernel_size_temp = round(((height * width) / (180 * 180)) ** 0.5)
    if kernel_size_temp % 2 == 0:
        kernel_size_temp = kernel_size_temp + 1
    kernel_size = max(5, kernel_size_temp)
    print(kernel_size)
    frame_2 = cv2.medianBlur(frame_1, kernel_size)
    cv2.imshow("medianBlur image", frame_2)
    cv2.imwrite('medianBlur image.jpeg', frame_2)
    # cv2.waitKey(0)
    print('frame_2', frame_2)

    # ret, binary_1 = cv2.threshold(frame_2, 100, 255, cv2.THRESH_BINARY)
    low_bound = 600
    upper_bound = 800
    frame_3 = cv2.Canny(frame_2, low_bound, upper_bound) # + cv2.Canny(s_channel, low_bound, upper_bound)
    cv2.imshow("Canny image", frame_3)
    cv2.imwrite('Canny image.jpeg', frame_3)
    # cv2.waitKey(0)

    '''contours, hierarchy = cv2.findContours(frame_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('contours', contours)
    print('contours', len(contours))
    # cv2.drawContours(frame_3, contours, 1, (0, 255, 0), 3)

    cv2.imshow("contours image", frame_3)
    cv2.waitKey(0)

    blank = np.zeros((height, width, 1), np.uint8)
    for contour in contours:
        contour_len = cv2.arcLength(contour, False)
        contour = cv2.approxPolyDP(contour, 0.01 * contour_len, False)
        print('contour', contour)
        cv2.drawContours(blank, contour, 1, (255, 0, 0), 3)
    cv2.imshow("contours after approxPolyDP image", blank)
    cv2.waitKey(0)'''


    lines_0 = cv2.HoughLinesP(frame_3, rho=1, theta=np.pi/180, threshold=round((height + width) / 30),
                              minLineLength=round((height + width) / 30), maxLineGap=round((height + width) / 30))
    # lines_1 = cv2.HoughLinesP(blank, rho=1, theta=np.pi / 180, threshold=round((height + width) / 400), minLineLength=round((height + width) / 40), maxLineGap=round((height + width) / 40))
    lines = []
    if lines_0 is not None:
        lines.extend(lines_0)
    #if lines_1 is not None:
    #    lines.extend(lines_1)
    print(type(lines_0))
    if lines is not []:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if (y1 >= height / 3) and (y2 >= height / 3):
                    cv2.line(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
                elif (y1 < height / 3) and (y2 >= height / 3):
                    y3 = height / 3
                    x3 = ((y3 - y1) / (y2 - y1)) * (x2 - x1) + x1
                    print(x3, y3)
                    cv2.line(frame, (round(x3), round(y3)), (x2, y2), color=(255, 0, 0), thickness=round((height + width) / 500))
                elif (y1 >= height / 3) and (y2 < height / 3):
                    y3 = height / 3
                    x3 = ((y3 - y1) / (y2 - y1)) * (x2 - x1) + x1
                    print(x3, y3)
                    cv2.line(frame, (x1, y1), (round(x3), round(y3)), color=(255, 0, 0), thickness=round((height + width) / 500))
    result = 1
    return result

###########################################################################

def runon_image(path) :
    frame = cv2.imread(path)
    height, width, channels = frame.shape
    print(frame.shape)
    if height > 1000 or width > 1000:
        factor = max(height, width)
        frame = cv2.resize(frame, None, fx=1000/factor, fy=1000/factor, interpolation=cv2.INTER_LINEAR)
        height, width, channels = frame.shape
    cv2.imshow("original image", frame)
    # cv2.waitKey(0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame.shape)
    cv2.imshow("RGB image", frame)
    cv2.imwrite('RGB image.jpeg', frame)
    # cv2.waitKey(0)

    detections_in_frame = detect_lane(frame, height, width)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("output image", frame)
    cv2.imwrite('output image.jpeg', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detections_in_frame

def runon_folder(path) :
    files = None
    if(path[-1] != "/"):
        path = path + "/"
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    all_detections = 0
    for f in files:
        print(f)
        f_detections = runon_image(f)
        all_detections += f_detections
    return all_detections

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', default='lane', help="requires path")
    args = parser.parse_args()
    folder = args.folder
    if folder is None :
        print("Folder path must be given \n Example: python proj1.py -folder images")
        sys.exit()

    if folder is not None :
        all_detections = runon_folder(folder)
        print("total of ", all_detections, " detections")

    cv2.destroyAllWindows()



