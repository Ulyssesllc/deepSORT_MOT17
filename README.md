# deepSORT_MOT17

## Mô tả

Đây là dự án thực hiện theo dõi đối tượng (Object Tracking) sử dụng các thuật toán SORT và DeepSORT trên bộ dữ liệu MOT17. Dự án bao gồm các notebook hướng dẫn, video kết quả và các file đánh giá hiệu suất.

## Cấu trúc thư mục

- `Lab04.ipynb`, `Lab04_ver2.ipynb`, `Lab04_ver3.ipynb`: Các notebook hướng dẫn thực nghiệm và phân tích kết quả.
- `deepsort_metrics.csv`, `sort_metrics.csv`, `sort_yolov8_metrics.csv`: Các file lưu trữ kết quả đánh giá hiệu suất của các thuật toán.
- `output_deepsort.mp4`, `output_sort.mp4`, `output_sort_yolov8.mp4`: Video kết quả theo dõi đối tượng.
- `Lab 04_ Object Tracking.pdf`: Tài liệu hướng dẫn lý thuyết và thực hành.

## Hướng dẫn sử dụng

1. Mở các notebook để xem và chạy từng bước thực nghiệm.
2. Xem các video để đánh giá trực quan kết quả theo dõi đối tượng.
3. Tham khảo các file CSV để so sánh hiệu suất các thuật toán.

## Yêu cầu

- Python
- Jupyter Notebook
- Các thư viện: numpy, opencv, matplotlib, v.v.

## Tác giả

- Ulyssesllc

## License

Dự án sử dụng cho mục đích học tập và nghiên cứu.

## Thống kê kết quả các thuật toán

| Thuật toán      | MOTA    | MOTP    | FP   | FN    | IDS | GT    | Matches |
|-----------------|---------|---------|------|-------|-----|-------|---------|
| SORT            | 0.2554  | 0.8853  | 1198 | 12613 | 25  | 18581 | 5968    |
| DeepSORT        | 0.2571  | 0.8874  | 1193 | 12582 | 29  | 18581 | 5999    |
| SORT + YOLOv8   | 0.2294  | 0.8127  | 778  | 13501 | 39  | 18581 | 5080    |

**Giải thích các chỉ số:**
- **MOTA**: Multi-Object Tracking Accuracy
- **MOTP**: Multi-Object Tracking Precision
- **FP**: False Positives
- **FN**: False Negatives
- **IDS**: Identity Switches
- **GT**: Ground Truth
- **Matches**: Số lần khớp đúng đối tượng
