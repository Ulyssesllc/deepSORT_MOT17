# vim: expandtab:ts=4:sw=4
import os
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Kiểm tra đường dẫn MOT17 và detections")
    p.add_argument(
        "--sequence_dir", required=True, help="VD: MOT17/train/MOT17-04-FRCNN"
    )
    p.add_argument(
        "--detection_file", required=True, help="VD: detections/MOT17-04-FRCNN.npy"
    )
    return p.parse_args()


def main():
    args = parse_args()
    seq = args.sequence_dir
    det = args.detection_file

    ok = True
    if not os.path.isdir(seq):
        print(f"[FAIL] Không tìm thấy sequence_dir: {seq}")
        ok = False
    else:
        img1 = os.path.join(seq, "img1")
        det_txt = os.path.join(seq, "det", "det.txt")
        print(f"[OK ] sequence_dir: {seq}")
        print(f"  - img1: {'OK' if os.path.isdir(img1) else 'MISSING'}")
        print(f"  - det/det.txt: {'OK' if os.path.exists(det_txt) else 'MISSING'}")

    if not os.path.exists(det):
        print(f"[FAIL] Không tìm thấy detection_file: {det}")
        ok = False
    else:
        print(f"[OK ] detection_file: {det}")

    if not ok:
        print("\nGợi ý:")
        print(
            "- Đặt dữ liệu theo cấu trúc: MOT17/train/<sequence>/{img1/, det/det.txt, (gt/gt.txt)}"
        )
        print(
            "- Tạo đặc trưng bằng: python tools/generate_detections.py --mot_dir MOT17/train --output_dir detections"
        )
        exit(1)

    print("\nMọi thứ sẵn sàng để chạy deep_sort_app.py")


if __name__ == "__main__":
    main()
