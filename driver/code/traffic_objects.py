# Quy tắc giao thông cùng cập nhật car_state: speed=0 yêu cầu dừng, speed_limit là giới hạn tốc độ.

from threading import Timer
import logging


# Lớp cơ sở cung cấp giao diện cập nhật trạng thái và phép kiểm tra độ gần.
class TrafficObject(object):

    def set_car_state(self, car_state):
        pass

    @staticmethod
    # Ước lượng độ gần bằng tỷ lệ chiều cao khung bao trên chiều cao ảnh, không đo khoảng cách thật.
    def is_close_by(obj, frame_height, min_height_pct=0.05):
        # Mặc định: khung bao cao hơn 5% chiều cao ảnh (ngưỡng 0.05).
        obj_height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
        return obj_height / frame_height > min_height_pct


# Đèn đỏ đủ gần: yêu cầu dừng bằng cách đặt speed=0.
class RedTrafficLight(TrafficObject):

    def set_car_state(self, car_state):
        logging.debug('red light: stopping car')
        car_state['speed'] = 0


# Đèn xanh giữ nguyên trạng thái, không hủy yêu cầu dừng từ vật thể khác.
class GreenTrafficLight(TrafficObject):

    def set_car_state(self, car_state):
        logging.debug('green light: make no changes')


# Người đi bộ đủ gần: yêu cầu dừng xe.
class Person(TrafficObject):

    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')

        car_state['speed'] = 0


# Chỉ cập nhật speed_limit, không ghi đè yêu cầu dừng trong speed.
class SpeedLimit(TrafficObject):

    def __init__(self, speed_limit):
        self.speed_limit = speed_limit

    def set_car_state(self, car_state):
        logging.debug('speed limit: set limit to %d' % self.speed_limit)
        car_state['speed_limit'] = self.speed_limit


# Ghi nhớ lần dừng: dừng một lần, chờ Timer rồi cho đi dù vẫn thấy biển.
# Mặc định chờ 3 giây và cần 20 lần không thấy biển liên tiếp để đặt lại trạng thái.
class StopSign(TrafficObject):
    """
    Stop Sign object would wait
    """

    def __init__(self, wait_time_in_sec=3, min_no_stop_sign=20):
        self.in_wait_mode = False
        self.has_stopped = False
        self.wait_time_in_sec = wait_time_in_sec
        self.min_no_stop_sign = min_no_stop_sign
        self.no_stop_count = min_no_stop_sign
        self.timer = None

    def set_car_state(self, car_state):
        self.no_stop_count = self.min_no_stop_sign

        if self.in_wait_mode:
            logging.debug('stop sign: 2) still waiting')
            # Giữ xe dừng trong thời gian chờ; mặc định 3 giây.
            car_state['speed'] = 0
            return

        if not self.has_stopped:
            logging.debug('stop sign: 1) just detected')

            car_state['speed'] = 0
            self.in_wait_mode = True
            self.has_stopped = True
            self.timer = Timer(self.wait_time_in_sec, self.wait_done)
            self.timer.start()
            return

    # Timer gọi hàm này sau thời gian chờ để kết thúc trạng thái chờ tại biển Stop.
    def wait_done(self):
        logging.debug('stop sign: 3) finished waiting for %d seconds' % self.wait_time_in_sec)
        self.in_wait_mode = False

    # Cho phép dừng ở biển tiếp theo sau đủ số lần liên tiếp không nhận diện thấy biển Stop.
    def clear(self):
        if self.has_stopped:
            # need this counter in case object detection has a glitch that one frame does not
            # detect stop sign, make sure we see 20 consecutive no stop sign frames (about 1 sec)
            # and then mark has_stopped = False
            self.no_stop_count -= 1
            if self.no_stop_count == 0:
                logging.debug("stop sign: 4) no more stop sign detected")
                self.has_stopped = False
                self.in_wait_mode = False  # may not need to set this
