import numpy as np
import json

# === Cấu hình đường dẫn ===
final_dataset_path = "processed_data/results_cnn_lstm/final_dataset.npz"
label_map_path = "processed_data/results_cnn_lstm/final_dataset_labels.json"

# Load dữ liệu
print("Đọc file final_dataset.npz ...")
data = np.load(final_dataset_path)
X = data["data"] # (num_samples, 16, 2048)
y = data["labels"] # (num_samples,)


print(f"Dữ liệu X shape: {X.shape}")
print(f"Nhãn y shape: {y.shape}")
print(f"Nhãn đầu tiên: {y[0]}")
print(f"Đặc trưng clip đầu tiên (X[0,0,:5]): {X[0, 0, :5]}")
print("")

# Thống kê nhãn
unique, counts = np.unique(y, return_counts=True)
print("Số lượng clip mỗi lớp:")
for label_id, count in zip(unique, counts):
    print(f"   Nhãn {label_id}: {count} clip")

# Đọc ánh xạ nhãn
with open(label_map_path, "r") as f:
    label_map = json.load(f)

print("\nÁnh xạ nhãn (label2id):")
for label_name, label_id in label_map.items():
    print(f"   {label_id}: {label_name}")
