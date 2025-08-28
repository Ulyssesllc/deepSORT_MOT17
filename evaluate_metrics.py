# vim: expandtab:ts=4:sw=4
import os
import argparse
from typing import Dict, List

import numpy as np
import motmetrics as mm


def iou_cost_matrix(
    objs: np.ndarray, hyps: np.ndarray, min_iou: float = 0.5
) -> np.ndarray:
    """Compute IoU-based cost matrix compatible with motmetrics.

    Boxes are in MOT format [x, y, w, h]. Returns a matrix with cost = 1 - IoU
    and np.nan where IoU < min_iou.
    """
    M = 0 if objs is None or objs.size == 0 else int(objs.shape[0])
    N = 0 if hyps is None or hyps.size == 0 else int(hyps.shape[0])

    if M == 0 or N == 0:
        # Return empty matrix with correct shape
        return np.empty((M, N))

    objs = objs.astype(np.float64, copy=False)
    hyps = hyps.astype(np.float64, copy=False)

    # Convert tlwh -> xyxy
    ox1 = objs[:, 0][:, None]
    oy1 = objs[:, 1][:, None]
    ox2 = (objs[:, 0] + objs[:, 2])[:, None]
    oy2 = (objs[:, 1] + objs[:, 3])[:, None]

    hx1 = hyps[:, 0][None, :]
    hy1 = hyps[:, 1][None, :]
    hx2 = (hyps[:, 0] + hyps[:, 2])[None, :]
    hy2 = (hyps[:, 1] + hyps[:, 3])[None, :]

    inter_w = np.clip(np.minimum(ox2, hx2) - np.maximum(ox1, hx1), a_min=0, a_max=None)
    inter_h = np.clip(np.minimum(oy2, hy2) - np.maximum(oy1, hy1), a_min=0, a_max=None)
    inter = inter_w * inter_h

    area_o = (ox2 - ox1) * (oy2 - oy1)
    area_h = (hx2 - hx1) * (hy2 - hy1)
    union = area_o + area_h - inter
    # Avoid division by zero
    iou = np.where(union > 0.0, inter / union, 0.0)

    cost = 1.0 - iou
    cost[iou < float(min_iou)] = np.nan
    return cost


def load_mot_files(gt_file: str, res_file: str):
    """
    Đọc dữ liệu GT và kết quả theo định dạng MOTChallenge (10 cột).
    Trả về hai mảng numpy với các cột [frame, id, x, y, w, h].
    """
    # GT: frame, id, x, y, w, h, mark, class, visibility
    gt = np.loadtxt(gt_file, delimiter=",")
    gt = gt[:, :6]  # lấy 6 cột đầu

    # Kết quả: frame, id, x, y, w, h, conf, -1, -1, -1
    res = np.loadtxt(res_file, delimiter=",")
    res = res[:, :6]
    return gt, res


def evaluate_split(mot_dir: str, result_dir: str, min_iou: float = 0.5):
    """
    Tính các độ đo MOTChallenge cho tất cả sequence có ground-truth trong mot_dir
    và có tệp kết quả tương ứng trong result_dir.
    """
    accs: Dict[str, mm.MOTAccumulator] = {}
    seq_names: List[str] = [
        d for d in os.listdir(mot_dir) if os.path.isdir(os.path.join(mot_dir, d))
    ]

    for seq in sorted(seq_names):
        seq_dir = os.path.join(mot_dir, seq)
        gt_file = os.path.join(seq_dir, "gt", "gt.txt")
        res_file = os.path.join(result_dir, f"{seq}.txt")
        if not (os.path.exists(gt_file) and os.path.exists(res_file)):
            print(f"[WARN] Bỏ qua {seq}: thiếu gt hoặc kết quả.")
            continue

        print(f"Đánh giá {seq} ...")
        gt, res = load_mot_files(gt_file, res_file)

        # Khởi tạo accumulator cho sequence
        acc = mm.MOTAccumulator(auto_id=True)

        # Tập khung hình cần duyệt
        frames = np.union1d(gt[:, 0].astype(int), res[:, 0].astype(int))
        for f in frames:
            gt_mask = gt[:, 0].astype(int) == f
            res_mask = res[:, 0].astype(int) == f

            gt_ids = gt[gt_mask, 1].astype(int).tolist()
            res_ids = res[res_mask, 1].astype(int).tolist()
            gt_boxes = gt[gt_mask, 2:6]
            res_boxes = res[res_mask, 2:6]

            if len(gt_ids) == 0 and len(res_ids) == 0:
                # Không có gì để cập nhật
                continue

            # Khoảng cách IoU tương thích NumPy 2.0 (tránh np.asfarray trong motmetrics)
            dists = iou_cost_matrix(gt_boxes, res_boxes, min_iou=min_iou)

            # Debug: kiểm tra format dữ liệu
            if dists.size == 0:
                # Nếu không có boxes, tạo ma trận trống
                dists = np.empty((len(gt_ids), len(res_ids)))
                if len(gt_ids) > 0 and len(res_ids) > 0:
                    dists.fill(np.nan)

            # Đảm bảo dists có đúng shape
            expected_shape = (len(gt_ids), len(res_ids))
            if dists.shape != expected_shape:
                print(
                    f"[WARN] Dists shape mismatch for frame {f}: expected {expected_shape}, got {dists.shape}"
                )
                # Tạo ma trận mới với shape đúng
                dists = np.empty(expected_shape)
                dists.fill(np.nan)

            # Chỉ update nếu có ít nhất một ID
            if len(gt_ids) > 0 or len(res_ids) > 0:
                try:
                    acc.update(gt_ids, res_ids, dists)
                except Exception as e:
                    print(f"[ERROR] Failed to update accumulator for frame {f}: {e}")
                    print(f"  gt_ids: {gt_ids}")
                    print(f"  res_ids: {res_ids}")
                    print(f"  dists shape: {dists.shape}")
                    continue

        accs[seq] = acc

    if not accs:
        raise SystemExit("Không có sequence hợp lệ để đánh giá.")

    # Validate accumulators before computing metrics
    valid_accs = {}
    for seq_name, acc in accs.items():
        if hasattr(acc, "events") and len(acc.events) > 0:
            valid_accs[seq_name] = acc
        else:
            print(f"[WARN] Skipping {seq_name}: empty accumulator")

    if not valid_accs:
        raise SystemExit("Không có accumulator hợp lệ để tính metrics.")

    mh = mm.metrics.create()
    try:
        # Use compute_many when available (newer motmetrics API). This avoids
        # passing a dict into compute(), which expects a DataFrame in newer versions.
        if hasattr(mh, "compute_many"):
            names = list(valid_accs.keys())
            acc_list = [valid_accs[n] for n in names]
            summary = mh.compute_many(
                acc_list, metrics=mm.metrics.motchallenge_metrics, names=names
            )
        else:
            summary = mh.compute(
                valid_accs, metrics=mm.metrics.motchallenge_metrics, name="MOT17"
            )
        str_summary = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
        )
    except Exception as e:
        print(f"[ERROR] Failed to compute metrics: {e}")
        print("Available sequences with data:")
        for seq_name, acc in valid_accs.items():
            print(f"  {seq_name}: {len(acc.events)} events")
        raise

    return summary, str_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Tính MOTA/MOTP/IDS cho MOT17")
    parser.add_argument(
        "--mot_dir", required=True, help="Thư mục split có gt, ví dụ MOT17/train"
    )
    parser.add_argument(
        "--result_dir", required=True, help="Thư mục chứa các kết quả <sequence>.txt"
    )
    parser.add_argument(
        "--output_csv", default=None, help="(Tùy chọn) Lưu bảng kết quả ra CSV"
    )
    parser.add_argument(
        "--min_iou",
        type=float,
        default=0.5,
        help="Ngưỡng IoU để ghép đối tượng (mặc định 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Thiết lập printer cho motmetrics (tùy chọn)
    mm.lap.default_solver = "lap"  # dùng lap solver mặc định nếu có

    summary, str_summary = evaluate_split(
        args.mot_dir, args.result_dir, min_iou=args.min_iou
    )
    print(str_summary)
    if args.output_csv is not None:
        # Lưu DataFrame ra CSV
        summary.to_csv(args.output_csv, index=True)
        print(f"Đã lưu kết quả: {args.output_csv}")


if __name__ == "__main__":
    main()
