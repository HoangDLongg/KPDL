import os
import numpy as np

# === Cấu hình đường dẫn ===
cnn_features_dir = "processed_data/results_cnn_lstm/cnn_features"

print("Kiểm tra một vài file đặc trưng CNN (.npy):\n")

# Duyệt 1 tư thế → 1 video → 1 clip → load file
for posture in os.listdir(cnn_features_dir):
    posture_path = os.path.join(cnn_features_dir, posture)
    for video in os.listdir(posture_path):
        video_path = os.path.join(posture_path, video)
        for clip_file in os.listdir(video_path):
            if clip_file.endswith(".npy"):
                file_path = os.path.join(video_path, clip_file)
                feature = np.load(file_path)
                
                print(f"File: {file_path}")
                print(f"Shape: {feature.shape} (expecting (16, 2048))")
                print(f"Frame 0 vector[:5]: {feature[0][:20]}") #khung thứ nhất (1/16),  20 đặc trưng đầu tiên (20/2048)
                print(f"Frame 1 vector[:5]: {feature[1][:20]}") #khung thứ hai, 20 đặc trưng đầu tiên
                print("-" * 60)
                break
        break
    break
