# Điểm khởi chạy DeepPiCar với tốc độ 40 (mức động cơ 0–100, không phải km/h).

from deep_pi_car import DeepPiCar
import logging
import sys

# Điểm vào chương trình khi chạy trực tiếp file này.
def main():
    # print system info
    logging.info('Starting DeepPiCar, system info: ' + sys.version)
    
    with DeepPiCar() as car:
        car.drive(40)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
