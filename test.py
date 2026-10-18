"""Desktop tester for the DeepPiCar lane and traffic-sign models.

Run with ``python test.py``. OpenCV handles the camera, image and video input;
Tkinter provides a small GUI so the file can be launched from an IDE as well.
"""

from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Thiếu thư viện cần thiết. Cài bằng lệnh: "
        "python -m pip install opencv-python numpy Pillow\n"
        f"Chi tiết: {exc}"
    )


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "object_detection" / "data" / "model_result" / "road_signs_quantized.tflite"
LABEL_PATH = ROOT_DIR / "models" / "object_detection" / "data" / "model_result" / "road_sign_labels.txt"
LANE_CODE_DIR = ROOT_DIR / "driver" / "code"
sys.path.insert(0, str(LANE_CODE_DIR))

try:
    from hand_coded_lane_follower import detect_lane
except ImportError as exc:
    raise SystemExit(f"Không nạp được mã nhận diện line: {exc}")


def make_interpreter(model_path):
    """Load a regular CPU TFLite interpreter, with either supported package."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            try:
                # Newer TensorFlow releases no longer re-export Interpreter
                # from tensorflow.lite, but retain it in this module.
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError as exc:
                raise RuntimeError(
                    "Để nhận diện biển báo, cần cài TensorFlow Lite: "
                    "py -3.13 -m pip install tensorflow\n"
                    "Trên Linux/Raspberry Pi có thể dùng gói nhẹ hơn: "
                    "python -m pip install tflite-runtime"
                ) from exc
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


class SignDetector:
    def __init__(self):
        if not MODEL_PATH.is_file() or not LABEL_PATH.is_file():
            raise RuntimeError(f"Không tìm thấy model hoặc nhãn biển báo trong {MODEL_PATH.parent}")
        self.interpreter = make_interpreter(MODEL_PATH)
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        shape = self.input_detail["shape"]
        self.height, self.width = int(shape[1]), int(shape[2])
        self.labels = {}
        with LABEL_PATH.open("r", encoding="utf-8") as labels_file:
            for line in labels_file:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    self.labels[int(parts[0])] = parts[1]

    def _outputs(self):
        values = []
        for detail in self.output_details:
            value = np.asarray(self.interpreter.get_tensor(detail["index"]))
            name = detail.get("name", "").lower()
            values.append((name, value))

        boxes = next((v for n, v in values if "box" in n and v.shape[-1:] == (4,)), None)
        scores = next((v for n, v in values if "score" in n), None)
        classes = next((v for n, v in values if "class" in n), None)

        # Some converted SSD models expose generic output names. Identify the
        # tensors by their shapes as a fallback.
        if boxes is None:
            boxes = next((v for _, v in values if v.ndim >= 2 and v.shape[-1] == 4), None)
        if scores is None or classes is None:
            vectors = [v for _, v in values if v.ndim >= 1 and v.size > 1 and v.shape[-1] != 4]
            vectors = [v for v in vectors if v.size > 1]
            if len(vectors) >= 2:
                scores = scores if scores is not None else vectors[-1]
                classes = classes if classes is not None else vectors[-2]
        if boxes is None or scores is None or classes is None:
            raise RuntimeError("Định dạng đầu ra của model TFLite không được hỗ trợ.")

        boxes = np.asarray(boxes).reshape(-1, 4)
        scores = np.asarray(scores).reshape(-1)
        classes = np.asarray(classes).reshape(-1)
        return boxes, classes, scores

    def detect(self, frame, threshold=0.30):
        tensor = cv2.resize(frame, (self.width, self.height))
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)[None, ...]
        dtype = self.input_detail["dtype"]
        if np.issubdtype(dtype, np.floating):
            tensor = tensor.astype(dtype) / 255.0
        elif dtype != np.uint8:
            scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
            if scale:
                tensor = (tensor / scale + zero_point).astype(dtype)
            else:
                tensor = tensor.astype(dtype)
        else:
            tensor = tensor.astype(dtype)

        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        boxes, classes, scores = self._outputs()
        height, width = frame.shape[:2]
        for box, class_id, score in zip(boxes, classes, scores):
            score = float(score)
            if score < threshold:
                continue
            # SSD detection boxes use [ymin, xmin, ymax, xmax], normalized 0..1.
            ymin, xmin, ymax, xmax = box
            if max(abs(float(box.min())), abs(float(box.max()))) <= 1.5:
                xmin, xmax = xmin * width, xmax * width
                ymin, ymax = ymin * height, ymax * height
            x1 = max(0, min(width - 1, int(xmin)))
            y1 = max(0, min(height - 1, int(ymin)))
            x2 = max(0, min(width - 1, int(xmax)))
            y2 = max(0, min(height - 1, int(ymax)))
            label = self.labels.get(int(class_id), f"Class {int(class_id)}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {score:.0%}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame


class DeepPiCarTester:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepPiCar - Test nhận diện")
        self.root.geometry("1000x740")
        self.capture = None
        self.after_id = None
        self.frame_delay_ms = 33
        self.photo = None
        self.sign_detector = None

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill="x")
        ttk.Label(controls, text="Chế độ:").pack(side="left")
        self.mode = tk.StringVar(value="Line trắng")
        ttk.Combobox(controls, textvariable=self.mode, state="readonly", width=18,
                     values=("Line trắng", "Biển báo", "Line + biển báo")).pack(side="left", padx=(5, 15))
        ttk.Label(controls, text="Camera index:").pack(side="left")
        self.camera_index = tk.StringVar(value="0")
        ttk.Spinbox(controls, from_=0, to=10, width=4, textvariable=self.camera_index).pack(side="left", padx=5)
        ttk.Button(controls, text="Mở camera", command=self.open_camera).pack(side="left", padx=4)
        ttk.Button(controls, text="Chọn ảnh", command=self.open_image).pack(side="left", padx=4)
        ttk.Button(controls, text="Chọn video", command=self.open_video).pack(side="left", padx=4)
        ttk.Button(controls, text="Dừng", command=self.stop).pack(side="left", padx=4)

        self.image_label = ttk.Label(root, anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=5)
        self.status = tk.StringVar(value="Chọn chế độ rồi mở camera, ảnh hoặc video.")
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _prepare_signs(self):
        if self.sign_detector is None:
            self.status.set("Đang nạp model nhận diện biển báo...")
            self.root.update_idletasks()
            self.sign_detector = SignDetector()

    def process_frame(self, frame):
        mode = self.mode.get()
        if mode in ("Line trắng", "Line + biển báo"):
            _, frame = detect_lane(frame)
        if mode in ("Biển báo", "Line + biển báo"):
            self._prepare_signs()
            frame = self.sign_detector.detect(frame)
        return frame

    def show_frame(self, frame):
        max_width, max_height = 960, 590
        scale = min(max_width / frame.shape[1], max_height / frame.shape[0], 1.0)
        if scale < 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.image_label.configure(image=self.photo)

    def open_camera(self):
        self.stop()
        try:
            index = int(self.camera_index.get())
        except ValueError:
            messagebox.showerror("Camera", "Camera index phải là số, ví dụ 0 hoặc 1.")
            return
        self._start_capture(index, "camera")

    def open_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[
            ("Ảnh", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Tất cả tệp", "*.*")])
        if not path:
            return
        self.stop()
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Ảnh", "Không đọc được ảnh đã chọn.")
            return
        self._process_and_show(frame, f"Ảnh: {Path(path).name}")

    def open_video(self):
        path = filedialog.askopenfilename(title="Chọn video", filetypes=[
            ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("Tất cả tệp", "*.*")])
        if not path:
            return
        self.stop()
        self._start_capture(path, "video")

    def _start_capture(self, source, source_type):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Nguồn video", "Không mở được camera/video. Hãy thử camera index khác hoặc kiểm tra tệp.")
            return
        if source_type == "video":
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            if 1 <= fps <= 120:
                self.frame_delay_ms = max(1, round(1000 / fps))
        else:
            self.frame_delay_ms = 33
        self.status.set(f"Đang chạy {source_type}. Nhấn Dừng để kết thúc.")
        self._update_video()

    def _update_video(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop()
            self.status.set("Đã phát hết video.")
            return
        if not self._process_and_show(frame, "Đang nhận diện..."):
            self.stop()
            return
        self.after_id = self.root.after(self.frame_delay_ms, self._update_video)

    def _process_and_show(self, frame, status):
        try:
            result = self.process_frame(frame)
            self.show_frame(result)
            self.status.set(status)
            return True
        except Exception as exc:
            messagebox.showerror("Lỗi nhận diện", str(exc))
            return False

    def stop(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    app_root = tk.Tk()
    app = DeepPiCarTester(app_root)
    app_root.mainloop()
