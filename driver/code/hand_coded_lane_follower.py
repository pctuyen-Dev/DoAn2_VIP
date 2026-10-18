# Bám làn bằng OpenCV: lọc màu → tìm biên → Hough → gộp vạch → tính góc lái.
# Tọa độ ảnh: x tăng sang phải, y tăng xuống dưới; góc servo 90 độ là đi thẳng.

import cv2
import numpy as np
import logging
import math
import datetime
import sys

_SHOW_IMAGE = False


class HandCodedLaneFollower(object):

    # car=None cho phép thử không có xe; góc ban đầu 90 độ là đi thẳng.
    def __init__(self, car=None):
        logging.info('Creating a HandCodedLaneFollower...')
        self.car = car
        self.curr_steering_angle = 90

    # Xử lý một khung hình để cập nhật góc lái và trả ảnh minh họa kết quả.
    def follow_lane(self, frame):
        # Main entry point of the lane follower
        show_image("orig", frame)

        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)

        return final_frame

    # Không có vạch thì giữ góc cũ; có vạch thì tính góc mới và hạn chế thay đổi đột ngột.
    def steer(self, frame, lane_lines):
        logging.debug('steering...')
        if len(lane_lines) == 0:
            logging.error('No lane lines detected, nothing to do.')
            return frame

        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        show_image("heading", curr_heading_image)

        return curr_heading_image


############################
# Các bước xử lý khung hình
############################
# Trả về tối đa hai vạch làn và ảnh đã vẽ vạch qua chuỗi bước xử lý bên dưới.
def detect_lane(frame):
    logging.debug('detecting lane lines...')

    edges = detect_edges(frame)
    show_image('edges', edges)

    cropped_edges = region_of_interest(edges)
    show_image('edges cropped', cropped_edges)

    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    show_image("line segments", line_segment_image)

    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    show_image("lane lines", lane_lines_image)

    return lane_lines, lane_lines_image


# Lọc vùng trắng trong HSV, rồi tìm biên bằng Canny.
def detect_edges(frame):
    # Vạch trắng thường có saturation thấp và value cao trong HSV.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image("hsv", hsv)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([179, 70, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    show_image("white mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges

# Hàm thử nghiệm cũ: quét 16 ngưỡng bão hòa; Canny dùng mặt nạ cuối vòng lặp.
def detect_edges_old(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image("hsv", hsv)
    for i in range(16):
        lower_blue = np.array([30, 16 * i, 0])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        show_image("blue mask Sat=%s" % (16* i), mask)


    #for i in range(16):
        #lower_blue = np.array([16 * i, 40, 50])
        #upper_blue = np.array([150, 255, 255])
        #mask = cv2.inRange(hsv, lower_blue, upper_blue)
       # show_image("blue mask hue=%s" % (16* i), mask)

        # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges


# Chỉ giữ nửa dưới ảnh, nơi có mặt đường gần xe.
def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # only focus bottom half of the screen

    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    show_image("mask", mask)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


# Hough xác suất tìm đoạn thẳng dài ít nhất 8 px, khoảng đứt tối đa 4 px.
def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)

    if line_segments is not None:
        # OpenCV versions may return (N, 4) or (N, 1, 4); normalize to (N, 1, 4)
        # because the downstream lane code iterates over one 4-coordinate row.
        line_segments = np.asarray(line_segments).reshape(-1, 1, 4)
        for line_segment in line_segments:
            logging.debug('detected line_segment:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments


# Gộp đoạn theo độ dốc và vị trí trái/phải; lấy trung bình hệ số mỗi nhóm.
# Bỏ đoạn thẳng đứng để tránh độ dốc vô hạn. Mỗi vạch có dạng [[x1, y1, x2, y2]].
def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # Đoạn vạch phải nằm trong 2/3 bên phải ảnh.

    for line_segment in line_segments:
        coords = np.asarray(line_segment).reshape(-1, 4)
        for x1, y1, x2, y2 in coords:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


# Một vạch: theo độ lệch hai đầu mút; hai vạch: hướng đến trung điểm hai đầu phía xa.
# atan(x_offset/y_offset) cho góc lệch với phương thẳng đứng; cộng 90 để đổi sang góc servo.
def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    logging.debug('new steering angle: %s' % steering_angle)
    return steering_angle


# Giới hạn đổi góc mỗi khung hình: mặc định 5 độ khi có hai vạch, 1 độ khi chỉ có một.
def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle


############################
# Các hàm tiện ích
############################
# Vẽ vạch trên lớp ảnh trống rồi trộn với ảnh gốc để quan sát.
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            coords = np.asarray(line).reshape(-1, 4)
            for x1, y1, x2, y2 in coords:
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


# Dùng lượng giác vẽ hướng đi từ giữa đáy ảnh lên nửa chiều cao; 90 độ là thẳng.
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


# Tính độ dài đoạn thẳng bằng khoảng cách Euclid giữa hai đầu mút.
def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Chỉ hiển thị ảnh khi cờ show được bật.
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


# Từ y=slope*x+intercept, lấy hai đầu mút ở đáy và giữa ảnh.
# Giới hạn x trong [-width, 2*width], nên điểm vẫn có thể nằm ngoài khung hình.
def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


############################
# Các hàm chạy thử
############################
# Thử bám làn trên một ảnh, không cần xe thật.
def test_photo(file):
    land_follower = HandCodedLaneFollower()
    frame = cv2.imread(file)
    combo_image = land_follower.follow_lane(frame)
    show_image('final', combo_image, True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Thử bám làn trên video và lưu ảnh gốc có góc lái trong tên cùng ảnh/video minh họa.
def test_video(video_file):
    lane_follower = HandCodedLaneFollower()
    cap = cv2.VideoCapture(video_file + '.avi')

    # Bỏ 3 khung hình đầu (không nhất thiết tương đương 1 giây).
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    video_overlay = cv2.VideoWriter("%s_overlay.avi" % (video_file), video_type, 20.0, (320, 240))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            print('frame %s' % i )
            combo_image= lane_follower.follow_lane(frame)
            
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower.curr_steering_angle), frame)
            
            cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), combo_image)
            video_overlay.write(combo_image)
            cv2.imshow("Road with Lane line", combo_image)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    test_video('/home/pi/DeepPiCar/driver/data/tmp/video01')
    #test_photo('/home/pi/DeepPiCar/driver/data/video/car_video_190427_110320_073.png')
    #test_photo(sys.argv[1])
    #test_video(sys.argv[1])
