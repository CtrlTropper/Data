# Hướng dẫn sử dụng Embedding Script

## Tổng quan
Script `embeding.py` đã được chỉnh sửa để có thể đọc và tạo embedding cho tất cả các tài liệu PDF trong thư mục `documents` và các thư mục con của nó.

## Cấu trúc thư mục
```
/workspace/
├── documents/
│   ├── Khac/           (26 file PDF)
│   └── LuatVietNam/    (37 file PDF)
├── Project/
│   ├── embeding.py     (Script chính)
│   ├── setup_environment.py
│   └── results/        (Thư mục kết quả sẽ được tạo tự động)
└── models/
    └── multilingual_e5_large/  (Model sẽ được tải tự động nếu không có)
```

## Các thay đổi chính

### 1. Xử lý đường dẫn
- ✅ Đã cập nhật từ `D:/Vian/Data/documents` thành `/workspace/documents`
- ✅ Đã cập nhật output directory thành `/workspace/Project/results`
- ✅ Hỗ trợ model path linh hoạt (local hoặc Hugging Face)

### 2. Xử lý thư mục con
- ✅ Sử dụng `os.walk()` để duyệt qua tất cả thư mục con
- ✅ Tìm tất cả file PDF trong `documents/Khac/` và `documents/LuatVietNam/`
- ✅ Tổng cộng: 63 file PDF

### 3. Cải thiện xử lý lỗi
- ✅ Kiểm tra file có thể đọc được không
- ✅ Xử lý lỗi cho từng file riêng biệt
- ✅ Thống kê chi tiết quá trình xử lý

### 4. Model loading linh hoạt
- ✅ Thử local model trước: `/workspace/models/multilingual_e5_large`
- ✅ Fallback về Hugging Face: `intfloat/multilingual-e5-large`
- ✅ Fallback cuối cùng: `all-MiniLM-L6-v2`

## Cách sử dụng

### 1. Cài đặt môi trường (nếu cần)
```bash
cd /workspace/Project
python3 setup_environment.py
```

### 2. Chạy script embedding
```bash
cd /workspace/Project
python3 embeding.py
```

## Kết quả đầu ra

### Cấu trúc thư mục kết quả:
```
/workspace/Project/results/
├── all_faiss.index          # FAISS index tổng hợp
├── all_embeddings.pkl       # Pickle data tổng hợp
└── [tên_file_pdf]/         # Thư mục riêng cho mỗi PDF
    ├── [tên_file]_ocr.txt
    ├── [tên_file]_clean.txt
    ├── [tên_file]_chunks.txt
    ├── [tên_file]_embeddings.pkl
    ├── [tên_file]_embedding_info.txt
    └── [tên_file]_faiss.index
```

### Thống kê xử lý:
- 📄 Tổng số file PDF: 63
- 📁 Thư mục Khac: 26 file
- 📁 Thư mục LuatVietNam: 37 file

## Tính năng mới

### 1. Xử lý hàng loạt thông minh
- Bỏ qua file đã được embedding trước đó
- Xử lý song song nhiều file
- Thống kê chi tiết progress

### 2. Lưu trữ có tổ chức
- Mỗi PDF có thư mục riêng
- File tổng hợp cho toàn bộ knowledge base
- Thông tin chi tiết về embeddings

### 3. Robust error handling
- Tiếp tục xử lý khi gặp lỗi với file cụ thể
- Log chi tiết các lỗi
- Báo cáo tổng kết cuối

## Yêu cầu hệ thống

### Thư viện cần thiết:
- PyMuPDF (fitz)
- pytesseract
- pillow (PIL)
- sentence-transformers
- transformers
- faiss-cpu
- underthesea
- torch
- numpy

### Phần cứng khuyến nghị:
- RAM: ít nhất 8GB
- GPU: CUDA-compatible (tùy chọn)
- Disk: ít nhất 5GB trống cho kết quả

## Troubleshooting

### 1. Lỗi import PyMuPDF
```bash
pip install PyMuPDF --break-system-packages
```

### 2. Lỗi CUDA
Script sẽ tự động fallback về CPU nếu không có GPU.

### 3. Lỗi model không tải được
Script sẽ tự động tải model từ Hugging Face hoặc sử dụng model nhỏ hơn.

### 4. Lỗi OCR
Đảm bảo tesseract được cài đặt:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-vie
```

## Liên hệ
Nếu có vấn đề, vui lòng kiểm tra log chi tiết trong quá trình chạy script.