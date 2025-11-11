import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import logging
import time

# Thiết lập logging riêng cho dữ liệu test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STEP3_EXTRACT_CNN_FEATURES_TEST] %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_test.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 (bỏ FC layer)
weights = ResNet50_Weights.DEFAULT
cnn_model = resnet50(weights=weights)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])
cnn_model.to(device).eval()

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Hàm trích đặc trưng 1 clip test
def extract_features_from_clip(clip_folder):
    start_time = time.time()
    logging.info(f"Bắt đầu trích đặc trưng từ clip test: {clip_folder}")
    features = []
    for i in range(16):
        img_path = os.path.join(clip_folder, f"frame_{i:02d}.jpg")
        if not os.path.exists(img_path):
            logging.error(f"Thiếu frame: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = cnn_model(img_tensor).squeeze().cpu().numpy()
        features.append(feat)
    if not features:
        logging.warning(f"Không có đặc trưng nào từ clip: {clip_folder}")
        return None
    features_array = np.stack(features)
    logging.info(f"Hoàn tất trích xuất {len(features)} đặc trưng trong {time.time() - start_time:.2f} giây")
    return features_array

# Duyệt toàn bộ folder clip test
def process_all_test_clips(input_dirs, output_dir):
    start_time = time.time()
    for input_dir, source_name in input_dirs:
        logging.info(f"Bắt đầu xử lý clip test từ {source_name}: {input_dir}")
        for label in os.listdir(input_dir):
            label_path = os.path.join(input_dir, label)
            if not os.path.isdir(label_path):
                continue
            for video_name in os.listdir(label_path):
                video_path = os.path.join(label_path, video_name)
                if not os.path.isdir(video_path):
                    continue
                for clip_folder in os.listdir(video_path):
                    clip_path = os.path.join(video_path, clip_folder)
                    out_path = os.path.join(output_dir, label, video_name, f"{clip_folder}.npy")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    features = extract_features_from_clip(clip_path)
                    if features is None:
                        continue
                    np.save(out_path, features)
                    logging.info(f"Đã lưu đặc trưng test tại: {out_path}")
        logging.info(f"Đã xong {source_name} trong {time.time() - start_time:.2f} giây")
    logging.info(f"Hoàn tất toàn bộ trích xuất đặc trưng test sau {time.time() - start_time:.2f} giây")

# Thực thi khi gọi trực tiếp
if __name__ == "__main__":
    input_dirs = [ 
        ("processed_data/results_cnn_lstm_test/cnn_lstm_clips_test", "clip test gốc"),
        ("processed_data/results_cnn_lstm_test/cnn_lstm_aug_test", "clip test tăng cường")
    ]
    output_dir = "processed_data/results_cnn_lstm_test/cnn_features_test"
    logging.info("Bắt đầu chương trình trích xuất đặc trưng cho TEST")
    process_all_test_clips(input_dirs, output_dir)
    logging.info("Trích xuất đặc trưng TEST hoàn tất.")
