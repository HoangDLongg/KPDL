import os
import cv2
import numpy as np
import random
from glob import glob
import logging
import time

# Thiết lập logging với bộ mã hóa UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STEP2_AUGMENT_CLIPS] %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Hàm lật ngang clip (dạng danh sách frame) 
def horizontal_flip(clip):
    return [cv2.flip(frame, 1) for frame in clip]

# Hàm điều chỉnh độ sáng ngẫu nhiên cho clip 
def adjust_brightness(clip, factor=0.2):
    return [np.clip(frame * (1 + random.uniform(-factor, factor)), 0, 1) for frame in clip]

# Hàm tăng cường clip bằng kỹ thuật đơn giản 
def augment_clip(clip):
    aug_clip = clip.copy()
    if random.random() < 0.5:
        aug_clip = horizontal_flip(aug_clip)
    if random.random() < 0.5:
        aug_clip = adjust_brightness(aug_clip)
    return aug_clip

# Hàm load 1 clip (16 ảnh) từ thư mục, resize và chuẩn hóa 
def load_clip(clip_folder):
    frame_paths = sorted(glob(os.path.join(clip_folder, "*.jpg")))
    logging.info(f"Tìm thấy {len(frame_paths)} khung hình trong {clip_folder}: {frame_paths}")  # Debug log
    if len(frame_paths) < 16:
        logging.warning(f"Bỏ qua {clip_folder}: chỉ có {len(frame_paths)} khung hình, cần tối thiểu 16")
        return []
    clip = [cv2.resize(cv2.imread(p), (224, 224)).astype(np.float32) / 255.0 for p in frame_paths]
    logging.info(f"Đã tải clip từ {clip_folder}")
    return clip

# Hàm lưu 1 clip (danh sách ảnh) vào thư mục 
def save_clip(clip, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for i, frame in enumerate(clip):
        out_path = os.path.join(out_folder, f"frame_{i:02d}.jpg")
        cv2.imwrite(out_path, (frame * 255).astype(np.uint8))
    logging.info(f"Đã lưu clip vào {out_folder}")

# Hàm tăng cường toàn bộ tập dữ liệu clip 
def augment_all_clips(input_root, output_root, aug_per_clip=2):
    start_time = time.time()
    logging.info(f"Bắt đầu tăng cường dữ liệu từ {input_root}")
    for label in os.listdir(input_root):
        label_path = os.path.join(input_root, label)
        if not os.path.isdir(label_path):
            logging.warning(f"Bỏ qua {label_path}: không phải thư mục")
            continue
        for video_name in os.listdir(label_path):
            video_path = os.path.join(label_path, video_name)
            if not os.path.isdir(video_path):
                logging.warning(f"Bỏ qua {video_path}: không phải thư mục")
                continue
            for clip_name in os.listdir(video_path):
                clip_path = os.path.join(video_path, clip_name)
                logging.info(f"Đang xử lý clip: {clip_path}")
                clip = load_clip(clip_path)
                if not clip:  # Bỏ qua nếu clip không hợp lệ
                    continue
                for i in range(aug_per_clip):        # Sinh nhiều phiên bản tăng cường từ 1 clip gốc
                    aug_clip = augment_clip(clip)
                    aug_name = f"{clip_name}_aug{i}"
                    save_path = os.path.join(output_root, label, video_name, aug_name)
                    save_clip(aug_clip, save_path)
    logging.info(f"Hoàn tất tăng cường dữ liệu trong {time.time() - start_time:.2f} giây")

# Chạy tăng cường nếu thực thi trực tiếp 
if __name__ == "__main__":
    input_root = "processed_data/results_cnn_lstm/cnn_lstm_clips"      # Clip gốc (chưa tăng cường)
    output_root = "processed_data/results_cnn_lstm/cnn_lstm_aug"       # Nơi lưu clip sau khi tăng cường
    logging.info("Bắt đầu chương trình tăng cường dữ liệu")
    augment_all_clips(input_root, output_root)
    logging.info("Tăng cường dữ liệu hoàn tất.")