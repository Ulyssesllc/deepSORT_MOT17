# deepSORT_MOT17

## Giới thiệu
Bài toán Multiple Object Tracking (MOT) nhằm gán ID nhất quán cho các đối tượng xuyên suốt chuỗi ảnh/video. Dự án này triển khai và so sánh:
- SORT: Theo dõi bằng Kalman Filter + gán ghép Hungarian theo IoU.
- DeepSORT: Bổ sung đặc trưng ReID (MARS) để tăng độ bền vững trước che khuất/va chạm.
- SORT + YOLOv8: Dùng YOLOv8 (COCO) làm detector thay vì file det có sẵn của MOT.

Dữ liệu: MOT17 (train/test), đánh giá theo các độ đo tiêu chuẩn: MOTA, MOTP, FP, FN, IDS.

## Cấu trúc thư mục
- Lab04.ipynb: SORT với detection sẵn (MOT det.txt).
- Lab04_ver2.ipynb: YOLOv8 + SORT (tự detect người).
- Lab04_ver3.ipynb: DeepSORT với embedding MARS.
- deepsort_metrics.csv, sort_metrics.csv, sort_yolov8_metrics.csv: Kết quả định lượng.
- output_deepsort.mp4, output_sort.mp4, output_sort_yolov8.mp4: Video minh họa.
- Lab 04_ Object Tracking.pdf: Tài liệu tham khảo.

## Cài đặt thuật toán
Yêu cầu: Python, Jupyter/Colab, các thư viện phổ biến.

Cài nhanh (nếu chạy local/Colab):
```bash
pip install -U pip opencv-python-headless numpy pandas scipy filterpy matplotlib tensorflow ultralytics
```

Các tham số chính (điều chỉnh ngay trong notebook, chạy 1 lần ra kết quả):
- Lab04.ipynb (SORT)
  - SEQ_ROOT: đường dẫn sequence (vd: /content/MOT17/train/MOT17-02-FRCNN)
  - DET_SCORE_THRESH: ngưỡng lọc detection từ det.txt (vd: 0.5)
  - ASSOC_IOU_THRESHOLD: IoU khi gán ghép (vd: 0.3)
  - EVAL_IOU_THRESHOLD: IoU khi đánh giá (vd: 0.5)
  - OUTPUT_VIDEO: đường dẫn video xuất ra (vd: /content/output_sort.mp4)
- Lab04_ver2.ipynb (YOLOv8 + SORT)
  - YOLO_MODEL: yolov8n/s/m.pt (vd: yolov8s.pt), YOLO_CONF (vd: 0.25), YOLO_IMG (vd: 640)
  - ASSOC_IOU_THRESHOLD, EVAL_IOU_THRESHOLD tương tự trên
  - OUTPUT_VIDEO, OUTPUT_TRACKS_TXT
- Lab04_ver3.ipynb (DeepSORT)
  - SEQ_ROOT, DET_SCORE_THRESH, ASSOC_IOU_THRESHOLD, EVAL_IOU_THRESHOLD
  - max_age, n_init, max_cosine_distance (vd: 30, 3, 0.2)
  - MARS model tự động tải về nếu chưa có: mars-small128.pb
  - OUTPUT_VIDEO: /content/output_deepsort.mp4

Quy trình chạy một lần:
1) Mở notebook tương ứng, chỉnh SEQ_ROOT/tham số. 2) Chạy toàn bộ cells theo thứ tự. 3) Sau khi chạy xong sẽ có video và file CSV metrics trong /content (hoặc thư mục làm việc).

## Kết quả thực nghiệm
- Định lượng (MOT17-02-FRCNN, tham số như trong notebook):
  | Thuật toán      | MOTA    | MOTP    | FP   | FN    | IDS | GT    | Matches |
  |-----------------|---------|---------|------|-------|-----|-------|---------|
  | SORT            | 0.2554  | 0.8853  | 1198 | 12613 | 25  | 18581 | 5968    |
  | DeepSORT        | 0.2571  | 0.8874  | 1193 | 12582 | 29  | 18581 | 5999    |
  | SORT + YOLOv8   | 0.2294  | 0.8127  | 778  | 13501 | 39  | 18581 | 5080    |

  Ghi chú độ đo:
  - MOTA: càng cao càng tốt (tổng hợp FP, FN, IDS).
  - MOTP: IoU trung bình giữa dự đoán và GT (càng cao càng tốt).
  - IDS: số lần đổi ID (càng thấp càng tốt).

- Định tính:
  - output_sort.mp4: Theo dõi ổn khi ít che khuất; dễ mất ID khi đám đông.
  - output_deepsort.mp4: ID ổn định hơn lúc che khuất ngắn; khôi phục ID tốt hơn.
  - output_sort_yolov8.mp4: Phụ thuộc chất lượng detect theo domain; có thể hụt người xa/nhỏ → FN tăng.

Các CSV metrics tương ứng đã được lưu: sort_metrics.csv, deepsort_metrics.csv, sort_yolov8_metrics.csv.

## Thảo luận
- Ưu điểm:
  - SORT: Rất nhanh, đơn giản, dễ tái lập. Phù hợp khi detector mạnh và bối cảnh ít che khuất.
  - DeepSORT: Thêm ReID giúp giảm nhầm lẫn ID, ổn hơn khi mục tiêu giao cắt/che khuất ngắn.
  - YOLOv8 + SORT: Linh hoạt khi không có det.txt; tận dụng detector hiện đại.
- Nhược điểm:
  - SORT: Nhạy với FN/FP của detector; dễ IDS khi mục tiêu gần nhau.
  - DeepSORT: Tốn tài nguyên (trích đặc trưng), phụ thuộc chất lượng embedding; nếu crop xấu hoặc quá nhỏ → ReID kém.
  - YOLOv8 + SORT: Hiệu năng phụ thuộc domain; nếu detector chưa phù hợp dữ liệu MOT → FN tăng, MOTA giảm.
- Hướng cải tiến khả thi:
  - Detector: Dùng YOLOv8x hoặc fine-tune theo domain MOT; điều chỉnh YOLO_CONF/NMS.
  - Tham số tracker: tăng DET_SCORE_THRESH (0.6–0.8) để giảm FP; tinh chỉnh max_age, n_init, max_cosine_distance (DeepSORT).
  - ReID: Dùng backbone mạnh hơn (OSNet, StrongSORT/OC-SORT/ByteTrack) hoặc huấn luyện lại ReID cho MOT.
  - Hợp nhất nhiều tiêu chí gán ghép (appearance + IoU + motion), bù chuyển động camera, xử lý tái xuất hiện ID.
  - Hậu xử lý: làm mượt quỹ đạo, loại bỏ track ngắn, hạ ngưỡng khởi tạo phù hợp với FPS.

## Kết luận
- DeepSORT cho MOTA và MOTP cao nhất trong thiết lập này, ID ổn định hơn, đặc biệt khi có che khuất ngắn.
- SORT vẫn là đường cơ sở nhanh, dễ dùng khi detector “sạch”.
- YOLOv8 + SORT hữu ích khi không có detection sẵn; cần tinh chỉnh/huấn luyện detector để nâng MOTA/MOTP.
- Tất cả notebook đã được chuẩn hoá để chạy một lần cho ra video và CSV đánh giá.

## Hướng dẫn sử dụng nhanh
1) Tải MOT17 về (vd: /content/MOT17 trên Colab).
2) Mở một trong các notebook, chỉnh SEQ_ROOT tới sequence mong muốn.
3) Run all. Kết quả: video (.mp4) + metrics (.csv) trong thư mục làm việc.

## Yêu cầu môi trường
- Python, Jupyter/Colab
- Thư viện: numpy, opencv, matplotlib, scipy, filterpy, pandas, tensorflow (DeepSORT), ultralytics (YOLOv8).

## Tác giả
- Ulyssesllc

## License
Dự án sử dụng cho mục đích học tập và nghiên cứu.
