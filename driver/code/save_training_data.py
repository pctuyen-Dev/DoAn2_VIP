# Tách video thành ảnh có nhãn góc lái do thuật toán OpenCV tạo ra.
# Nhãn nằm trong tên ảnh, không phải góc lái do người gán.

import cv2
import sys
from hand_coded_lane_follower import HandCodedLaneFollower


# Nhận đường dẫn chưa có đuôi .avi; lưu ảnh theo mẫu <video>_<số thứ tự>_<góc lái>.
def save_image_and_steering_angle(video_file):
    lane_follower = HandCodedLaneFollower()
    cap = cv2.VideoCapture(video_file + '.avi')

    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            lane_follower.follow_lane(frame)
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower.curr_steering_angle), frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    save_image_and_steering_angle(sys.argv[1])
