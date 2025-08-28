# Deep SORT cho MOT17

Theo dõi đa đối tượng (Multi-Object Tracking) với Deep SORT trên bộ dữ liệu MOT17. Repo này đã được chỉnh để chạy trơn tru với chuẩn MOT17, xuất kết quả theo định dạng MOT, trực quan hóa và đánh giá MOTA/MOTP/IDS.

## 1) Cài đặt nhanh
- Yêu cầu: Python 3.8+, pip, (tuỳ chọn) GPU với CUDA cho TensorFlow nếu muốn tăng tốc trích xuất đặc trưng.
- Cài đặt (Windows bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
# Hoặc GPU: pip install -r requirements-gpu.txt
```

## 2) Tải và chuẩn bị dữ liệu MOT17 (quan trọng)
Cấu trúc chuẩn (như bạn đã liệt kê) phải đúng như sau:
```
MOT17/
  train/
    MOT17-02-DPM/
      img1/ (N ảnh, tên dạng 000001.jpg ...)
      det/det.txt
      gt/gt.txt
      seqinfo.ini
    MOT17-04-FRCNN/
      img1/, det/det.txt, gt/gt.txt, seqinfo.ini
    ... tất cả các sequence train khác ...
  test/
    MOT17-01-FRCNN/
      img1/
      det/det.txt
      seqinfo.ini
    ... (lưu ý: test không có gt/gt.txt)
```
- Script `tools/generate_detections.py` sẽ duyệt các thư mục con có `img1/` và `det/det.txt`; bỏ qua thư mục không hợp lệ.
- Script `evaluate_motchallenge.py` sẽ bỏ qua sequence nếu thiếu file `.npy` trong `detections/`.

- Đặt thư mục `MOT17` cùng cấp với mã nguồn repo này.
- (Tuỳ chọn) Kiểm tra đường dẫn trước khi chạy:
```bash
python validate_paths.py \
  --sequence_dir MOT17/train/MOT17-04-FRCNN \
  --detection_file detections/MOT17-04-FRCNN.npy
```

## 3) Tạo đặc trưng ReID cho detections
Tạo `.npy` đặc trưng cho từng sequence (bắt buộc trước khi tracking):

```bash
python tools/generate_detections.py \
  --mot_dir MOT17/train \
  --output_dir detections
# (Tuỳ chọn cho test)
# python tools/generate_detections.py --mot_dir MOT17/test --output_dir detections
```
- Tham số `--model` mặc định `resources/networks/mars-small128.pb`.

## 4) Chạy Deep SORT
### 4.1 Chạy toàn bộ split
```bash
python evaluate_motchallenge.py \
  --mot_dir MOT17/train \
  --detection_dir detections \
  --output_dir results \
  --min_confidence 0.3 \
  --nms_max_overlap 1.0 \
  --max_cosine_distance 0.2 \
  --nn_budget 100
```

### 4.2 Chạy 1 sequence đơn lẻ
```bash
python deep_sort_app.py \
  --sequence_dir MOT17/train/MOT17-04-FRCNN \
  --detection_file detections/MOT17-04-FRCNN.npy \
  --output_file results/MOT17-04-FRCNN.txt \
  --min_confidence 0.3 \
  --nms_max_overlap 1.0 \
  --max_cosine_distance 0.2 \
  --nn_budget 100 \
  --display True
```

## 5) Chạy trên Google Colab (đường dẫn ví dụ)
```bash
python tools/generate_detections.py \
  --mot_dir /content/deepSORT_MOT17/MOT17/train \
  --output_dir /content/deepSORT_MOT17/detections

python deep_sort_app.py \
  --sequence_dir /content/deepSORT_MOT17/MOT17/train/MOT17-04-FRCNN \
  --detection_file /content/deepSORT_MOT17/detections/MOT17-04-FRCNN.npy \
  --output_file /content/deepSORT_MOT17/results/MOT17-04-FRCNN.txt \
  --min_confidence 0.3 --nms_max_overlap 1.0 \
  --max_cosine_distance 0.2 --nn_budget 100 \
  --display False
```

## 6) Xem kết quả và xuất video
- Xem kết quả:
```bash
python show_results.py \
  --sequence_dir MOT17/train/MOT17-04-FRCNN \
  --result_file results/MOT17-04-FRCNN.txt \
  --detection_file detections/MOT17-04-FRCNN.npy \
  --update_ms 20
```
- Xuất video và (tuỳ chọn) chuyển mã H.264:
```bash
python generate_videos.py \
  --mot_dir MOT17/train \
  --result_dir results \
  --output_dir videos \
  --update_ms 20 \
  --convert_h264
```

## 7) Đánh giá MOTA/MOTP/IDS
Dùng script `evaluate_metrics.py` (cần `motmetrics`):
```bash
python evaluate_metrics.py \
  --mot_dir MOT17/train \
  --result_dir results \
  --output_csv mot17_train_metrics.csv \
  --min_iou 0.5
```

## 8) Tham số quan trọng
- `--min_confidence`, `--nms_max_overlap`, `--max_cosine_distance`, `--nn_budget`, `--min_detection_height`.

## 9) Khắc phục sự cố
- FileNotFoundError cho img1 hoặc detection_file: kiểm tra cấu trúc thư mục và chạy `validate_paths.py`.
- Lỗi `cv2`/`numpy`: cài đặt từ `requirements.txt`, kích hoạt venv.
- TensorFlow không tương thích: dùng TF 1.x hoặc TF 2.x với `tf.compat.v1`.
- FPS hiển thị: dùng `--update_ms` hoặc kiểm tra `seqinfo.ini`.

## 10) Trích dẫn

- Deep SORT:
  - N. Wojke, A. Bewley, D. Paulus, "Simple Online and Realtime Tracking with a Deep Association Metric," 2017 IEEE International Conference on Image Processing (ICIP), 2017.
  - arXiv: https://arxiv.org/abs/1703.07402

  - N. Wojke, A. Bewley, "Deep Cosine Metric Learning for Person Re-Identification," 2018 IEEE Winter Conference on Applications of Computer Vis

- MOTChallenge (MOT17): https://motchallenge.net

## 11) Giấy phép
Xem `LICENSE`.

Lưu ý model ReID (bắt buộc):
- Cần tệp `resources/networks/mars-small128.pb`. Nếu chưa có, hãy tải và đặt đúng vị trí trên.
- Hoặc truyền đường dẫn tuyệt đối qua tham số `--model` khi chạy `tools/generate_detections.py`.
- Trên Colab, ví dụ:
```bash
python tools/generate_detections.py \
  --model /content/deepSORT_MOT17/resources/networks/mars-small128.pb \
  --mot_dir /content/deepSORT_MOT17/MOT17/train \
  --output_dir /content/deepSORT_MOT17/detections
```
