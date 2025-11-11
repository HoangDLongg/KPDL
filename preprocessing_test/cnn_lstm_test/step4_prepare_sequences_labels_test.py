import os
import numpy as np
import json
import logging
import time

# Thiết lập logging với bộ mã hóa UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STEP4_PREPARE_SEQUENCES_LABELS_TEST] %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Hàm tổng hợp tất cả đặc trưng và nhãn thành tập dữ liệu duy nhất
def prepare_dataset(feature_dir, output_npz):
    start_time = time.time()
    logging.info(f"Bắt đầu tổng hợp dữ liệu test từ đặc trưng tại: {feature_dir}")
    
    data = []
    labels = []

    # Lấy danh sách tên các lớp (label) và ánh xạ label -> id
    try:
        label_names = sorted([d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))])
    except Exception as e:
        logging.error(f"Không thể đọc thư mục feature_dir: {e}")
        return

    if not label_names:
        logging.warning("Không tìm thấy lớp nào trong thư mục đầu vào.")
        return

    label2id = {label: idx for idx, label in enumerate(label_names)}
    logging.info(f"Tìm thấy {len(label_names)} lớp: {label2id}")

    # Duyệt qua từng file .npy trong mỗi lớp
    for label in label_names:
        label_path = os.path.join(feature_dir, label)
        if not os.path.isdir(label_path):
            logging.warning(f"Bỏ qua {label_path}: không phải thư mục")
            continue
        for root, _, files in os.walk(label_path):
            for file in files:
                if file.endswith(".npy"):
                    fpath = os.path.join(root, file)
                    try:
                        features = np.load(fpath)
                        data.append(features)
                        labels.append(label2id[label])
                    except Exception as e:
                        logging.error(f"Lỗi khi load file {fpath}: {e}")

    # Kiểm tra và log số lượng mẫu
    if len(data) > 0:
        logging.info(f"Tổng cộng: {len(data)} mẫu, shape mỗi mẫu: {data[0].shape}")
    else:
        logging.warning("Không có dữ liệu hợp lệ nào để lưu.")
        return

    # Chuyển thành mảng numpy
    try:
        data = np.array(data)
        labels = np.array(labels)
    except Exception as e:
        logging.error(f"Lỗi khi chuyển dữ liệu sang mảng numpy: {e}")
        return

    # Lưu dữ liệu vào file .npz
    try:
        np.savez(output_npz, data=data, labels=labels)
        logging.info(f"Đã lưu tập dữ liệu test vào: {output_npz}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file npz: {e}")
        return

    # Lưu label mapping ra file JSON
    label_json_path = output_npz.replace(".npz", "_labels.json")
    try:
        with open(label_json_path, "w", encoding="utf-8") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=2)
        logging.info(f"Đã lưu label mapping tại: {label_json_path}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file JSON: {e}")
        return

    logging.info(f"Hoàn tất tổng hợp dữ liệu test trong {time.time() - start_time:.2f} giây")

# Thực thi nếu chạy trực tiếp
if __name__ == "__main__":
    input_dir = "processed_data/results_cnn_lstm_test/cnn_features_test"
    output_npz = "processed_data/results_cnn_lstm_test/final_dataset_test.npz"
    logging.info("Bắt đầu chương trình tổng hợp dữ liệu TEST và nhãn")
    prepare_dataset(input_dir, output_npz)
    logging.info("Tổng hợp dữ liệu TEST hoàn tất.")
