# Deep SORT cho MOT17

Theo dõi đa đối tượng (Multi-Object Tracking) với Deep SORT trên bộ dữ liệu MOT17. Repo này đã được chỉnh để chạy trơn tru với chuẩn MOT17, xuất kết quả theo định dạng MOT, trực quan hóa và đánh giá MOTA/MOTP/IDS.

## 1) Cài đặt nhanh
- Yêu cầu: Python 3.8+, pip, (tuỳ chọn) GPU với CUDA cho TensorFlow nếu muốn tăng tốc trích xuất đặc trưng.
- Cài đặt (Windows bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
# Hoặc dùng GPU: pip install -r requirements-gpu.txt
```

Lưu ý: Module trích xuất đặc trưng dùng TensorFlow (tf.compat.v1). Có thể dùng TF 1.x hoặc TF 2.x với compat v1.

## 2) Chuẩn bị dữ liệu MOT17
Đặt dữ liệu với cấu trúc:
```
MOT17/
  train/
    MOT17-02-FRCNN/
      img1/, det/det.txt, gt/gt.txt, seqinfo.ini
    ...
  test/
    MOT17-01-FRCNN/
      img1/, det/det.txt, seqinfo.ini
    ...
```
Thư mục `MOT17` nên nằm cùng cấp với mã nguồn (như repo này).

## 3) Tạo đặc trưng ReID cho detections
Tạo `.npy` đặc trưng cho từng sequence (bắt buộc trước khi tracking):

```bash
python tools/generate_detections.py \
  --mot_dir MOT17/train \
  --output_dir detections
# (Tuỳ chọn cho test)
# python tools/generate_detections.py --mot_dir MOT17/test --output_dir detections
```

- Tham số `--model` mặc định `resources/networks/mars-small128.pb`. Hãy tải và cập nhật đường dẫn nếu cần.

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
Kết quả dạng MOT được lưu vào `results/<sequence>.txt`.

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

## 5) Xem kết quả và xuất video
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

## 6) Đánh giá MOTA/MOTP/IDS
Dùng script `evaluate_metrics.py` (cần `motmetrics`):
```bash
python evaluate_metrics.py \
  --mot_dir MOT17/train \
  --result_dir results \
  --output_csv mot17_train_metrics.csv \
  --min_iou 0.5
```
In bảng tổng hợp ra terminal và (nếu chỉ định) lưu CSV.

## 7) Tham số quan trọng
- `--min_confidence`: ngưỡng tin cậy phát hiện (ảnh hưởng FP/FN).
- `--nms_max_overlap`: ngưỡng NMS cho boxes chồng lấn.
- `--max_cosine_distance`: ngưỡng matching theo đặc trưng ReID (ảnh hưởng IDS).
- `--nn_budget`: số đặc trưng lưu trữ mỗi ID (bộ nhớ/độ mượt).
- `--min_detection_height`: loại phát hiện quá nhỏ.

## 8) Khắc phục sự cố
- Lỗi `ImportError: cv2`/`numpy`: đảm bảo cài đặt requirements và kích hoạt venv.
- Thiếu `detections/<sequence>.npy`: chạy bước tạo đặc trưng (mục 3).
- TensorFlow không tương thích: dùng TF 1.x hoặc TF 2.x với `tf.compat.v1`.
- FPS hiển thị: dùng `--update_ms` hoặc kiểm tra `seqinfo.ini`.

## 9) Trích dẫn

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }


## 10) Giấy phép
Xem `LICENSE`.
