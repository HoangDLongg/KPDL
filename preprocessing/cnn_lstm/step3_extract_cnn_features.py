import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import logging
import time

# Thiết lập logging với bộ mã hóa UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STEP3_EXTRACT_CNN_FEATURES] %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Cấu hình thiết bị: dùng GPU nếu có, không thì dùng CPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình ResNet50 và bỏ lớp Fully Connected để lấy đặc trưng 
weights = ResNet50_Weights.DEFAULT
cnn_model = resnet50(weights=weights)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])  # Bỏ FC layer
cnn_model.to(device).eval() # Đưa model về chế độ đánh giá

# Tiền xử lý ảnh: resize và chuyển thành tensor 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Hàm trích xuất đặc trưng từ 1 clip gồm 16 frame 
def extract_features_from_clip(clip_folder):
    start_time = time.time()
    logging.info(f"Bắt đầu trích xuất đặc trưng từ clip: {clip_folder}")
    features = []
    for i in range(16):
        img_path = os.path.join(clip_folder, f"frame_{i:02d}.jpg")
        if not os.path.exists(img_path):
            logging.error(f"Không tìm thấy ảnh: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = cnn_model(img_tensor).squeeze().cpu().numpy() # Trích đặc trưng
        features.append(feat)
    if not features:
        logging.warning(f"Không trích xuất được đặc trưng nào từ {clip_folder}")
        return None
    features_array = np.stack(features) # Kết quả: mảng (16, 2048)
    logging.info(f"Trích xuất {len(features)} đặc trưng từ {clip_folder} trong {time.time() - start_time:.2f} giây")
    return features_array

# Duyệt toàn bộ cấu trúc thư mục: label -> video -> clip từ cả hai nguồn
def process_all_clips(input_dirs, output_dir):
    start_time = time.time()
    for input_dir, source_name in input_dirs:
        logging.info(f"Bắt đầu xử lý clip từ {source_name}: {input_dir}")
        for label in os.listdir(input_dir):
            label_path = os.path.join(input_dir, label)
            if not os.path.isdir(label_path):
                logging.warning(f"Bỏ qua {label_path}: không phải thư mục")
                continue
            for video_name in os.listdir(label_path):
                video_path = os.path.join(label_path, video_name)
                if not os.path.isdir(video_path):
                    logging.warning(f"Bỏ qua {video_path}: không phải thư mục")
                    continue
                for clip_folder in os.listdir(video_path):
                    clip_path = os.path.join(video_path, clip_folder)
                    out_path = os.path.join(output_dir, label, video_name, f"{clip_folder}.npy")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    logging.info(f"Trích xuất đặc trưng từ ({source_name}): {clip_path}")
                    features = extract_features_from_clip(clip_path)
                    if features is None:
                        logging.warning(f"Bỏ qua lưu file do không có đặc trưng: {out_path}")
                        continue
                    np.save(out_path, features)
                    logging.info(f"Đã lưu đặc trưng tại: {out_path}")
        logging.info(f"Hoàn tất xử lý clip từ {source_name} trong {time.time() - start_time:.2f} giây")
    logging.info(f"Hoàn tất xử lý toàn bộ clip trong {time.time() - start_time:.2f} giây")

# Chạy toàn bộ quá trình nếu thực thi trực tiếp
if __name__ == "__main__":
    input_dirs = [ 
        ("processed_data/results_cnn_lstm/cnn_lstm_clips", "clip gốc"),
        ("processed_data/results_cnn_lstm/cnn_lstm_aug", "clip tăng cường")
    ]
    output_dir = "processed_data/results_cnn_lstm/cnn_features"
    logging.info(f"Bắt đầu chương trình trích xuất đặc trưng từ {len(input_dirs)} nguồn")
    process_all_clips(input_dirs, output_dir)
    logging.info("Trích xuất đặc trưng hoàn tất.")