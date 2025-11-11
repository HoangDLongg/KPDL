import os
import cv2
import numpy as np
from glob import glob
import logging
import time

# Thiết lập logging với bộ mã hóa UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STEP1_CLIP_SLICING] %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Cấu hình số frame mỗi clip, bước trượt và kích thước frame 
FRAMES_PER_CLIP = 16  # 16 frame liên tiếp / clip
STEP = 8              
FRAME_SIZE = (224, 224)

# Hàm tạo clip từ các frame ảnh
def create_clips_from_frames(input_dir, output_dir, verbose=True):
    start_time = time.time()
    frame_paths = sorted(glob(os.path.join(input_dir, "*.jpg")))
    total_frames = len(frame_paths)

    if total_frames < FRAMES_PER_CLIP:
        logging.warning(f"Bỏ qua {input_dir}: chỉ có {total_frames} khung hình, cần tối thiểu {FRAMES_PER_CLIP}")
        return 0

    clip_idx = 0
    for start in range(0, total_frames - FRAMES_PER_CLIP + 1, STEP):
        clip_frames = frame_paths[start:start + FRAMES_PER_CLIP]
        clip_folder = os.path.join(output_dir, f"clip_{clip_idx:03d}")
        os.makedirs(clip_folder, exist_ok=True)
        for i, frame_path in enumerate(clip_frames):
            img = cv2.imread(frame_path)
            img = cv2.resize(img, FRAME_SIZE)
            img = img.astype(np.float32) / 255.0
            out_path = os.path.join(clip_folder, f"frame_{i:02d}.jpg")
            cv2.imwrite(out_path, (img * 255).astype(np.uint8))
        clip_idx += 1
    if verbose:
        logging.info(f"Đã tạo {clip_idx} clip từ {input_dir} trong {time.time() - start_time:.2f} giây")
    return clip_idx

# Hàm xử lý toàn bộ video trong thư mục
def process_all_videos(root_input_dir, root_output_dir, verbose=True):
    start_time = time.time()
    total_clips = 0
    for label in os.listdir(root_input_dir):
        label_path = os.path.join(root_input_dir, label)
        if not os.path.isdir(label_path):
            continue
        for video_folder in os.listdir(label_path):
            input_dir = os.path.join(label_path, video_folder)
            output_dir = os.path.join(root_output_dir, label, video_folder)
            os.makedirs(output_dir, exist_ok=True)
            if verbose:
                logging.info(f"Đang xử lý: {input_dir}")
            clip_count = create_clips_from_frames(input_dir, output_dir, verbose)
            total_clips += clip_count
    if verbose:
        logging.info(f"Hoàn tất tạo {total_clips} clip trong {time.time() - start_time:.2f} giây")

if __name__ == "__main__":
    process_all_videos(
        "processed_data/frames", "processed_data/results_cnn_lstm/cnn_lstm_clips", verbose=True)