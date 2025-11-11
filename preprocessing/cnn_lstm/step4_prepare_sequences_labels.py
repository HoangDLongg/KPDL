import os
import numpy as np
import json

# Hàm tổng hợp tất cả đặc trưng và nhãn thành tập dữ liệu duy nhất 
def prepare_dataset(feature_dir, output_npz):
    data = []
    labels = []
    label_names = sorted(os.listdir(feature_dir)) # Lấy tên các lớp     
    label2id = {label: idx for idx, label in enumerate(label_names)} # Ánh xạ tên -> id


    # Duyệt qua từng clip đã trích đặc trưng, gom thành 1 tập duy nhất
    for label in label_names:
        label_path = os.path.join(feature_dir, label)
        for root, _, files in os.walk(label_path):
            for file in files:
                if file.endswith(".npy"):
                    fpath = os.path.join(root, file)
                    features = np.load(fpath)
                    data.append(features)
                    labels.append(label2id[label])

    # Chuyển danh sách thành mảng numpy: (num_samples, 16, 2048)
    data = np.array(data)
    labels = np.array(labels)

    # Lưu tập huấn luyện vào file .npz (nén)
    np.savez(output_npz, data=data, labels=labels)

    # Lưu mapping nhãn ra file json (phục vụ training và inference)
    with open(output_npz.replace(".npz", "_labels.json"), "w") as f:
        json.dump(label2id, f)
    print(f"Lưu dữ liệu vào {output_npz} và label map.")

# Chạy toàn bộ quá trình nếu thực thi trực tiếp 
if __name__ == "__main__":
    prepare_dataset("processed_data/results_cnn_lstm/cnn_features", "processed_data/results_cnn_lstm/final_dataset.npz")
