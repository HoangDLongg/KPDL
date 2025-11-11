import cv2
import os
import logging
import time

# Thiết lập logging với bộ mã hóa UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PREPROCESS_COMMON] %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', mode='a', encoding='utf-8'),  # Sử dụng UTF-8 cho file log
        logging.StreamHandler()  # Sử dụng console mặc định, sẽ được cấu hình lại nếu cần
    ]
)

# Hàm trích xuất frame từ 1 video, lưu thành ảnh JPG theo FPS mong muốn
def extract_frames_from_video(video_path, output_folder, fps=2, resize=(224, 224), start_time=None):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if not video_fps or video_fps == 0:
        logging.error(f"Lỗi khi đọc FPS từ video: {video_path}")
        return

    interval = int(video_fps / fps)
    frame_count = 0
    saved_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            h, w = frame.shape[:2]

            # CROP chính xác vùng chứa người tập
            # Cắt bỏ phần đầu (logo zdsoft), viền đen dưới, bên phải (máy tính)
            top = int(0.18 * h)        # bỏ nhiều hơn phía trên (từ 12% → 18%)
            bottom = int(0.90 * h)     # giữ vừa đủ dưới
            left = int(0.20 * w)       # bỏ nhiều hơn bên trái (từ 18% → 22%)
            right = int(0.80 * w)      # bỏ nhiều hơn bên phải (từ 88% → 85%)

            frame = frame[top:bottom, left:right]
        
            # Resize sau khi crop
            if resize:
                frame = cv2.resize(frame, resize)

            out_path = os.path.join(output_folder, f"frame_{saved_frame:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_frame += 1
        frame_count += 1
    cap.release()
    if start_time is not None:
        logging.info(f"Đã lưu {saved_frame} frame từ {video_path} trong {time.time() - start_time:.2f} giây")

# Hàm xử lý toàn bộ video trong thư mục, phân theo nhãn (label)
def prepare_all_videos(root_dir='datasetset/train', out_dir='processed_data/frames', fps=2):
    start_time = time.time()
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for video_file in os.listdir(label_dir):
            if not video_file.endswith('.mp4'):
                continue
            video_path = os.path.join(label_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(out_dir, label, video_name)
            logging.info(f"Đang xử lý: {video_path}")
            extract_frames_from_video(video_path, output_folder, fps=fps, start_time=start_time)
    logging.info(f"Hoàn tất trích xuất toàn bộ video trong {time.time() - start_time:.2f} giây")

# Chạy toàn bộ quá trình nếu thực thi trực tiếp
if __name__ == "__main__":
    prepare_all_videos(
        root_dir='dataset/train',
        out_dir='processed_data/frames',
        fps=2
    )
