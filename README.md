# Deep SORT cho MOT17

Theo dõi đa đối tượng (MOT) bằng Deep SORT trên bộ dữ liệu MOT17. Repo đã chuẩn hoá luồng xử lý: tạo đặc trưng ReID cho detections → chạy tracker → trực quan hoá → xuất video → đánh giá MOTA/MOTP/IDS.

## 1) Cài đặt
- Yêu cầu: Python 3.8+, pip. GPU (tùy chọn) để tăng tốc trích xuất đặc trưng.
- Windows bash (khuyến nghị dùng venv):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
# Nếu có GPU: pip install -r requirements-gpu.txt
```

## 2) Dữ liệu MOT17
Đặt thư mục `MOT17` cùng cấp với mã nguồn repo này theo đúng cấu trúc:
```
MOT17/
  train/
    MOT17-02-DPM/
      img1/, det/det.txt, gt/gt.txt, seqinfo.ini
    MOT17-04-FRCNN/
      img1/, det/det.txt, gt/gt.txt, seqinfo.ini
    ...
  test/
    MOT17-01-FRCNN/
      img1/, det/det.txt, seqinfo.ini
    ... (test không có gt/gt.txt)
```
Lưu ý: Test không có ground-truth; chỉ dùng để chạy/ghi kết quả.

## 3) Mô hình ReID (bắt buộc)
- Cần tệp: `resources/networks/mars-small128.pb`.
- Nếu chưa có, hãy tải về và đặt đúng đường dẫn trên; hoặc truyền đường dẫn tuyệt đối qua `--model` khi tạo đặc trưng.

Ví dụ (Colab):
```bash
python tools/generate_detections.py \
  --model /content/deepSORT_MOT17/resources/networks/mars-small128.pb \
  --mot_dir /content/deepSORT_MOT17/MOT17/train \
  --output_dir /content/deepSORT_MOT17/detections
```

## 4) Tạo đặc trưng ReID cho detections (.npy)
Tạo cho train (và tùy chọn cho test):
```bash
python tools/generate_detections.py \
  --mot_dir MOT17/train \
  --output_dir detections
# Tuỳ chọn cho test
# python tools/generate_detections.py --mot_dir MOT17/test --output_dir detections
```

## 5) Chạy Deep SORT
- Chạy toàn bộ split:
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
- Chạy 1 sequence:
```bash
python deep_sort_app.py \
  --sequence_dir MOT17/train/MOT17-04-FRCNN \
  --detection_file detections/MOT17-04-FRCNN.npy \
  --output_file results/MOT17-04-FRCNN.txt \
  --min_confidence 0.3 \
  --nms_max_overlap 1.0 \
  --max_cosine_distance 0.2 \
  --nn_budget 100 \
  --display False
```

## 6) Trực quan hoá và xuất video
- Xem kết quả (không GUI trên Colab):
```bash
python show_results.py \
  --sequence_dir MOT17/train/MOT17-04-FRCNN \
  --result_file results/MOT17-04-FRCNN.txt \
  --detection_file detections/MOT17-04-FRCNN.npy \
  --update_ms 20
```
- Xuất video (.mp4) và tuỳ chọn chuyển mã H.264:
```bash
python generate_videos.py \
  --mot_dir MOT17/train \
  --result_dir results \
  --output_dir videos \
  --update_ms 20 \
  --convert_h264
```

## 7) Đánh giá (train)
Chỉ áp dụng cho train (có GT):
```bash
python evaluate_metrics.py \
  --mot_dir MOT17/train \
  --result_dir results \
  --output_csv mot17_train_metrics.csv \
  --min_iou 0.5
```

## 8) Kiểm tra đường dẫn
Sau khi đã tạo `.npy`:
```bash
python validate_paths.py \
  --sequence_dir MOT17/train/MOT17-04-FRCNN \
  --detection_file detections/MOT17-04-FRCNN.npy
```

## 9) Mẹo chạy trên Colab
- Đặt `MOT17/` vào trong đường dẫn repo (ví dụ `/content/deepSORT_MOT17/MOT17`).
- Cài thêm: `tensorflow`, `tf-keras`, `tf-slim` nếu cần.
- Dùng `--display False` khi chạy tracker để tránh lỗi GUI.
- Dùng notebook `Demonstration.ipynb` để xem demo đầu-cuối và xem video inline.

## 10) Khắc phục sự cố
- NotFoundError mars-small128.pb: đặt file vào `resources/networks/` hoặc truyền `--model /đường/dẫn/tới/.pb`.
- FileNotFoundError img1/det: kiểm tra cấu trúc MOT17 và chạy `validate_paths.py` (sau khi đã tạo `.npy`).
- Lỗi import TF/CV/NumPy: kích hoạt venv và cài `requirements.txt` (hoặc `requirements-gpu.txt`).
- Không có kết quả ở test khi đánh giá: test không có GT, chỉ đánh giá được trên train.

## 11) Trích dẫn
Nếu sử dụng repo trong báo cáo/bài viết, vui lòng trích dẫn:
- N. Wojke, A. Bewley, D. Paulus, "Simple Online and Realtime Tracking with a Deep Association Metric," ICIP 2017. arXiv: https://arxiv.org/abs/1703.07402
- MOTChallenge (MOT17): https://motchallenge.net

## 12) Giấy phép
Xem `LICENSE`.
