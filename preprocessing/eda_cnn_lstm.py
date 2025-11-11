# eda_cnn_lstm.py

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# === Load dữ liệu và ánh xạ nhãn ===
npz_path = "processed_data/results_cnn_lstm/final_dataset.npz"
json_path = "processed_data/results_cnn_lstm/final_dataset_labels.json"

data = np.load(npz_path)
X = data["data"]             # shape: (N, 16, 2048)
y = data["labels"]           # shape: (N,)

with open(json_path, encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# === THỐNG KÊ CƠ BẢN ===
print("= THỐNG KÊ DỮ LIỆU SAU TIỀN XỬ LÝ CNN+LSTM:")
print(f"Số mẫu (clips): {X.shape[0]}")
print(f"Số frame mỗi clip: {X.shape[1]}")
print(f"Kích thước đặc trưng mỗi frame: {X.shape[2]}")
print(f"Tổng chiều dài vector đặc trưng / clip: {X.shape[1] * X.shape[2]}")

unique_labels, counts = np.unique(y, return_counts=True)
print("\n= Phân bố số mẫu theo nhãn:")
for label_id, count in zip(unique_labels, counts):
    print(f"   - {id2label[label_id]:<15}: {count} clips")

# === MẪU DỮ LIỆU ĐẦU TIÊN ===
print("\n= Ví dụ mẫu:")
print(f"   - Nhãn: {id2label[y[0]]} (id={y[0]})")
print(f"   - Vector đặc trưng frame đầu tiên: {X[0,0,:5]}... (5 giá trị đầu)")

# === TRỰC QUAN PHÂN BỐ NHÃN ===
label_series = pd.Series(y).map(id2label)
plt.figure(figsize=(8, 5))
sns.countplot(x=label_series, order=label_series.value_counts().index)
plt.title("Biểu đồ phân bố số lượng mẫu theo tư thế yoga")
plt.xlabel("Tư thế")
plt.ylabel("Số clip")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_label_distribution.png")
plt.show()

# === THỐNG KÊ GIÁ TRỊ ĐẶC TRƯNG ===
X_flat = X.reshape(X.shape[0], -1)
print("\n= ĐẶC TRƯNG TOÀN BỘ CLIP:")
print(f"   - Shape sau khi flatten: {X_flat.shape}")
print(f"   - Giá trị trung bình: {X_flat.mean():.4f}")
print(f"   - Độ lệch chuẩn:       {X_flat.std():.4f}")
print(f"   - Min: {X_flat.min():.4f} | Max: {X_flat.max():.4f}")

# === PCA VÀ BIỂU ĐỒ 2D ===
print("\n= Thực hiện PCA (2 thành phần)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)

plt.figure(figsize=(8, 6))
for label_id in np.unique(y):
    idx = y == label_id
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=id2label[label_id], alpha=0.6)
plt.title("Biểu đồ PCA 2D – Phân bố các clip sau tiền xử lý")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("eda_pca_2d.png")
plt.show()
