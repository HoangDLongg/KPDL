import os
import cv2
import numpy as np
import random
from glob import glob
import logging
import time

# Thiết lập logging riêng cho test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STEP2_AUGMENT_CLIPS_TEST] %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_test.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Hàm lật ngang clip
def horizontal_flip(clip):
    return [cv2.flip(frame, 1) for frame in clip]

# Hàm điều chỉnh độ sáng ngẫu nhiên
def adjust_brightness(clip, factor=0.2):
    return [np.clip(frame * (1 + random.uniform(-factor, factor)), 0, 1) for frame in clip]

# Hàm tăng cường dữ liệu cho 1 clip
def augment_clip(clip):
    aug_clip = clip.copy()
    if random.random() < 0.5:
        aug_clip = horizontal_flip(aug_clip)
    if random.random() < 0.5:
        aug_clip = adjust_brightness(aug_clip)
    return aug_clip

# Hàm load clip từ folder
def load_clip(clip_folder):
    frame_paths = sorted(glob(os.path.join(clip_folder, "*.jpg")))
    logging.info(f"Tìm thấy {len(frame_paths)} khung hình trong {clip_folder}: {frame_paths}")
    if len(frame_paths) < 16:
        logging.warning(f"Bỏ qua {clip_folder}: chỉ có {len(frame_paths)} khung hình, cần tối thiểu 16")
        return []
    clip = [cv2.resize(cv2.imread(p), (224, 224)).astype(np.float32) / 255.0 for p in frame_paths]
    return clip

# Hàm lưu clip
def save_clip(clip, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for i, frame in enumerate(clip):
        out_path = os.path.join(out_folder, f"frame_{i:02d}.jpg")
        cv2.imwrite(out_path, (frame * 255).astype(np.uint8))
    logging.info(f"Đã lưu clip vào {out_folder}")

# Hàm tăng cường toàn bộ clip test
def augment_all_test_clips(input_root, output_root, aug_per_clip=2):
    start_time = time.time()
    logging.info(f"Bắt đầu tăng cường dữ liệu test từ {input_root}")
    for label in os.listdir(input_root):
        label_path = os.path.join(input_root, label)
        if not os.path.isdir(label_path):
            continue
        for video_name in os.listdir(label_path):
            video_path = os.path.join(label_path, video_name)
            if not os.path.isdir(video_path):
                continue
            for clip_name in os.listdir(video_path):
                clip_path = os.path.join(video_path, clip_name)
                logging.info(f"Đang xử lý clip test: {clip_path}")
                clip = load_clip(clip_path)
                if not clip:
                    continue
                for i in range(aug_per_clip):
                    aug_clip = augment_clip(clip)
                    aug_name = f"{clip_name}_aug{i}"
                    save_path = os.path.join(output_root, label, video_name, aug_name)
                    save_clip(aug_clip, save_path)
    logging.info(f"Hoàn tất tăng cường dữ liệu test trong {time.time() - start_time:.2f} giây")

# Gọi hàm khi chạy trực tiếp
if __name__ == "__main__":
    input_root = "processed_data/results_cnn_lstm_test/cnn_lstm_clips_test"    # Clip test gốc
    output_root = "processed_data/results_cnn_lstm_test/cnn_lstm_aug_test"     # Clip test sau tăng cường
    logging.info("Bắt đầu chương trình tăng cường dữ liệu cho TEST")
    augment_all_test_clips(input_root, output_root)
    logging.info("Tăng cường dữ liệu TEST hoàn tất.")
