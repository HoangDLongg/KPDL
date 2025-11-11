# 🧘 Dự án Nhận diện Tư thế Yoga (CNN-LSTM)

Dự án này sử dụng mô hình CNN-LSTM để phân loại 5 tư thế yoga (Bhujasana, Padamasana, Tadasana, Trikasana, Vrikshasana) từ dữ liệu video.

* **CNN (ResNet50):** Được sử dụng như một bộ trích xuất đặc trưng, chuyển đổi mỗi khung hình video thành một vector 2048 chiều.
* **LSTM:** Được sử dụng để học mối quan hệ thời gian (temporal patterns) giữa chuỗi 16 khung hình liên tiếp.

Mục tiêu là xây dựng một hệ thống nhận diện tư thế tự động với độ chính xác cao, đạt **96.88** trên tập dữ liệu kiểm tra.

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3
* **Deep Learning:** PyTorch
* **Trích xuất đặc trưng:** ResNet50 (từ TorchVision)
* **Xử lý Video/Ảnh:** OpenCV
* **Khoa học dữ liệu:** NumPy, Pandas, Scikit-learn (PCA, metrics)
* **Trực quan hóa:** Matplotlib, Seaborn

---

## 🚀 Luồng chạy dự án (Pipeline)

Dự án bao gồm 2 giai đoạn: Tiền xử lý (để chuẩn bị file `.npz`) và Huấn luyện (để đào tạo mô hình).

### Giai đoạn 1: Tiền xử lý (Preprocessing)

Các script phải được chạy theo thứ tự sau để xử lý cả tập train và test:

**1. Trích xuất Frames (Video -> Frames)**
Trích xuất khung hình (2 FPS), crop và resize từ video `.mp4` gốc.

```bash
# Xử lý tập train
python preprocess_common.py
# Xử lý tập test
python preprocess_common_test.py
```

**2. Cắt Clips (Frames -> Clips) Tổ chức các frame thành các clip 16-frame theo kiểu cửa sổ trượt (bước trượt 8 frame).**
```bash
python step1_clip_slicing.py
```
**3. Tăng cường dữ liệu (Clips -> Aug Clips) Tạo thêm dữ liệu huấn luyện bằng cách lật ngang và thay đổi độ sáng.**
```bash
python step2_augment_clips.py
```
**4.Trích xuất Đặc trưng (Clips -> Features) Dùng ResNet50 để biến mỗi clip (16, 224, 224, 3) thành vector đặc trưng (16, 2048).**
```bash
python step3_extract_cnn_features.py
```
**5. Tổng hợp Dữ liệu (Features -> NPZ) Gom tất cả các file đặc trưng .npy thành một file .npz nén duy nhất.**
```bash
python step4_prepare_sequences_labels.py
```
### Giai đoạn 2: Huấn luyện (Training)

Sau khi có file final_dataset.npz và final_dataset_test.npz, mở và chạy file train.ipynb để:

**1 Tải dữ liệu đã xử lý.**

Đầu vào: Script đọc file final_dataset.npz (tạo ra từ Giai đoạn 1).

Kết quả: Nó nạp dữ liệu vào 2 biến:

train_features: Mảng NumPy kích thước (1635, 16, 2048), (1635 clip, 16 frame/clip, 2048 đặc trưng/frame).

train_labels: Mảng NumPy kích thước (1635,) chứa nhãn (0-4) cho mỗi clip.

**2 Định nghĩa và huấn luyện mô hình LSTMClassifier.**
Script định nghĩa một lớp (class) tên là LSTMClassifier.

Đây chính là "bộ não" của mô hình. Kiến trúc của nó rất quan trọng:

Lớp LSTM: Nó nhận đầu vào là chuỗi (16, 2048) (16 frame, mỗi frame 2048 đặc trưng). Nhiệm vụ của lớp này là "đọc" tuần tự 16 frame để tìm ra mối liên hệ thời gian (temporal patterns) giữa chúng.

Lớp FC (Linear): Lớp này nhận đầu ra của LSTM (trạng thái ẩn cuối cùng) và "ép" nó thành 5 đầu ra, tương ứng với 5 tư thế yoga.

**3. Vòng lặp Huấn luyện (Training Loop)**
Đây là phần cốt lõi, được định nghĩa trong hàm train_and_evaluate. Quá trình này lặp đi lặp lại (ví dụ: 20 Epochs):

Lấy một "lô" (Batch): Tải 32 clip (batch_size=32) từ dữ liệu huấn luyện.

Dự đoán (Forward Pass): Đưa 32 clip này vào mô hình LSTMClassifier. Mô hình sẽ dự đoán (ví dụ: "Clip 1 là 'Tadasana', Clip 2 là 'Bhujasana',...").

Tính Lỗi (Calculate Loss): So sánh dự đoán của mô hình với nhãn thực tế (train_labels). Nó dùng CrossEntropyLoss để tính xem mô hình đã dự đoán "sai" đến mức nào.

Học hỏi (Backward Pass): Dựa trên mức độ "sai", nó tính toán ngược lại (Backpropagation) và dùng Adam Optimizer để "tinh chỉnh" trọng số của mô hình LSTM, để lần dự đoán tiếp theo sẽ chính xác hơn.

Lặp lại: Lặp lại quá trình này cho đến khi hết dữ liệu huấn luyện.

Kết quả của 20 epoch cho thấy Loss giảm từ 0.2518 xuống 0.0000, nghĩa là mô hình đã "học thuộc" tập train rất tốt.
**4 Đánh giá mô hình và xem kết quả.**
Sau khi "học" xong, mô hình được mang đi "thi" trên tập test (dữ liệu nó chưa bao giờ thấy):

Mô hình dự đoán trên toàn bộ 407 mẫu của tập test.

Kết quả dự đoán được so sánh với nhãn thật.

Kết quả: Mô hình đạt độ chính xác 96.89%.
