# %pip install langchain langchain-community langchain-openai
# !pip install underthesea

# Try different PyMuPDF import methods
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import PyMuPDF as fitz
    except ImportError:
        try:
            import pymupdf as fitz
        except ImportError:
            print("❌ Không thể import PyMuPDF. Vui lòng cài đặt: pip install PyMuPDF")
            exit(1)
# Import with error handling
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import io
    from datetime import datetime
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import pickle
    import os
    import re
    from underthesea import sent_tokenize
    from transformers import AutoTokenizer
    import faiss
    import torch
    print("✅ Đã import thành công tất cả thư viện cần thiết!")
except ImportError as e:
    print(f"❌ Lỗi import thư viện: {e}")
    print("💡 Vui lòng cài đặt các thư viện cần thiết:")
    print("   pip install PyMuPDF pytesseract pillow sentence-transformers transformers faiss-cpu underthesea torch numpy")
    exit(1)

# Kiểm tra GPU
print(f"Số lượng GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Tổng VRAM: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024**2:.2f} GB")

device = torch.device("cuda:0")
print(f"\nĐang sử dụng: {torch.cuda.get_device_name(device)}")

# Load model embedding
# Try local path first, then fallback to Hugging Face
local_model_path = "/workspace/models/multilingual_e5_large"
huggingface_model_name = "intfloat/multilingual-e5-large"

try:
    if os.path.exists(local_model_path):
        print(f"📦 Sử dụng model local: {local_model_path}")
        model = SentenceTransformer(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        print(f"📦 Model local không tồn tại, tải từ Hugging Face: {huggingface_model_name}")
        model = SentenceTransformer(huggingface_model_name)
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
        print("✅ Đã tải model thành công từ Hugging Face")
except Exception as e:
    print(f"❌ Lỗi khi tải model: {e}")
    print("🔄 Thử sử dụng model mặc định: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Đường dẫn input folder chứa các PDF (mới thêm để xử lý hàng loạt)
input_folder = "/workspace/documents"  # Updated to workspace documents folder
output_dir = "/workspace/Project/results"
os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục output nếu chưa có

all_faiss_path = os.path.join(output_dir, "all_faiss.index")
all_pickle_path = os.path.join(output_dir, "all_embeddings.pkl")

def is_pdf_embedded(path):
    """
    Kiểm tra xem file PDF đã được embedding hay chưa dựa trên file pickle chung.
    """
    if not os.path.exists(all_pickle_path):
        return False
    pdf_name = os.path.splitext(os.path.basename(path))[0]
    with open(all_pickle_path, 'rb') as f:
        all_data = pickle.load(f)
    existing_pdf_names = {entry['pdf_name'] for entry in all_data}
    return pdf_name in existing_pdf_names

def preprocess_image(img):
    """
    Tiền xử lý ảnh để cải thiện OCR.
    """
    if img.mode != 'L':
        img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_pdf_to_text(pdf_path, output_dir):
    """
    OCR file PDF thành text và lưu vào folder riêng.
    """
    try:
        print(f"📖 Đang OCR file: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        full_text = ""
        ocr_config = r'--oem 3 --psm 6 -l vie'
        for page_num in range(total_pages):
            print(f"🔄 Xử lý trang {page_num + 1}/{total_pages}...")
            page = doc.load_page(page_num)
            matrix = fitz.Matrix(2.5, 2.5)
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img = preprocess_image(img)
            page_text = pytesseract.image_to_string(img, config=ocr_config)
            full_text += page_text.strip()
        doc.close()
        print(f"✅ Hoàn thành OCR {total_pages} trang")
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_folder = os.path.join(output_dir, pdf_name)
        os.makedirs(pdf_folder, exist_ok=True)
        output_path = os.path.join(pdf_folder, f"{pdf_name}_ocr.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"📄 Kết quả OCR lưu vào: {output_path}")
        return full_text
    except Exception as e:
        print(f"❌ Lỗi OCR cho file {pdf_path}: {e}")
        return None

def clean_text(text, pdf_path, output_dir):
    """
    Làm sạch text OCR và lưu vào folder riêng.
    """
    text = re.sub(r'[^\w\s.,;:()\[\]?!\"\'\-–—…°%‰≥≤→←≠=+/*<>\n\r]', '', text)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n(?=\w)', ' ', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    cleaned_text = text.strip()
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_folder = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_folder, exist_ok=True)
    output_path = os.path.join(pdf_folder, f"{pdf_name}_clean.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"📄 Kết quả clean lưu vào: {output_path}")
    return cleaned_text

def split_sections(text):
    """
    Tách text thành sections theo tiêu đề.
    """
    sections = re.split(r'\n(?=(?:[IVXLCDM]+\.)|(?:\d+\.)|(?:[a-z]\)))', text)
    return [s.strip() for s in sections if s.strip()]

def split_text_to_chunks_vi_tokenized_with_section(text, chunk_size=512, overlap=50):
    """
    Chia text thành chunks dựa trên token, giữ cấu trúc section.
    """
    sections = split_sections(text)
    all_chunks = []
    for section in sections:
        sentences = sent_tokenize(section)
        current_chunk = []
        current_tokens = 0
        for sentence in sentences:
            num_tokens = len(tokenizer.tokenize(sentence))
            if current_tokens + num_tokens > chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                all_chunks.append(chunk_text)
                overlap_chunk = []
                total = 0
                for s in reversed(current_chunk):
                    toks = len(tokenizer.tokenize(s))
                    if total + toks > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    total += toks
                current_chunk = overlap_chunk + [sentence]
                current_tokens = total + num_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += num_tokens
        if current_chunk:
            all_chunks.append(' '.join(current_chunk).strip())
    return all_chunks

def create_embeddings(chunks):
    """
    Tạo embeddings cho chunks.
    """
    try:
        print(f"🔄 Tạo embeddings cho {len(chunks)} chunks...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        print(f"✅ Hoàn thành tạo embeddings")
        return embeddings
    except Exception as e:
        print(f"❌ Lỗi tạo embeddings: {e}")
        return None

def save_embeddings(chunks, embeddings, pdf_path, output_dir):
    """
    Lưu embeddings vào folder riêng và cập nhật file chung.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_folder = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_folder, exist_ok=True)

    data = {
        'pdf_name': pdf_name,
        'chunks': chunks,
        'embeddings': embeddings,
        'created_at': datetime.now().isoformat()
    }

    # Lưu embeddings pickle riêng
    pickle_path = os.path.join(pdf_folder, f"{pdf_name}_embeddings.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    # Lưu chunks text riêng
    chunks_path = os.path.join(pdf_folder, f"{pdf_name}_chunks.txt")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        f.write(f"CHUNKS TỪ FILE: {pdf_name}.pdf\n")
        f.write(f"Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tổng số chunks: {len(chunks)}\n")
        f.write("=" * 60 + "\n\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"CHUNK {i}:\n")
            f.write("-" * 30 + "\n")
            f.write(chunk + "\n")
            f.write("-" * 30 + "\n\n")

    # Lưu info embeddings riêng
    embedding_info_path = os.path.join(pdf_folder, f"{pdf_name}_embedding_info.txt")
    with open(embedding_info_path, 'w', encoding='utf-8') as f:
        f.write(f"THÔNG TIN EMBEDDINGS: {pdf_name}.pdf\n")
        f.write(f"Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"📊 THỐNG KÊ:\n")
        f.write(f"- Tổng số chunks: {len(chunks)}\n")
        f.write(f"- Kích thước embeddings: {embeddings.shape}\n")
        f.write(f"- Kiểu dữ liệu: {embeddings.dtype}\n")
        f.write(f"- Kích thước mỗi vector: {embeddings.shape[1]} dimensions\n\n")
        f.write(f"📝 PREVIEW EMBEDDINGS (5 chunks đầu):\n")
        f.write("-" * 50 + "\n")
        for i in range(min(5, len(chunks))):
            f.write(f"\nCHUNK {i+1}:\n")
            f.write(f"Text: {chunks[i][:100]}...\n")
            f.write(f"Embedding vector (10 giá trị đầu): {embeddings[i][:10].tolist()}\n")
            f.write(f"Vector norm: {np.linalg.norm(embeddings[i]):.4f}\n")
            f.write("-" * 30 + "\n")

    # Lưu FAISS index riêng
    index_path = os.path.join(pdf_folder, f"{pdf_name}_faiss.index")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    # Cập nhật FAISS chung
    if os.path.exists(all_faiss_path):
        index_all = faiss.read_index(all_faiss_path)
    else:
        index_all = faiss.IndexFlatL2(dim)
    index_all.add(embeddings.astype(np.float32))
    faiss.write_index(index_all, all_faiss_path)

    # Cập nhật pickle chung
    if os.path.exists(all_pickle_path):
        with open(all_pickle_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        all_data = []
    all_data.append(data)
    with open(all_pickle_path, 'wb') as f:
        pickle.dump(all_data, f)

    print(f"💾 Đã lưu embeddings riêng: {pickle_path}")
    print(f"📄 Đã lưu chunks riêng: {chunks_path}")
    print(f"📊 Đã lưu info embeddings riêng: {embedding_info_path}")
    print(f"📌 Đã lưu FAISS index riêng: {index_path}")
    print(f"🔁 Cập nhật FAISS chung: {all_faiss_path}")
    print(f"📦 Cập nhật pickle chung: {all_pickle_path}")

    return pickle_path, index_path

# Quy trình chính: Embedding tất cả PDF trong input_folder và các thư mục con (chức năng mới)
def get_all_pdf_files(folder_path):
    """
    Lấy tất cả file PDF từ folder và các thư mục con.
    """
    pdf_files = []
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục không tồn tại: {folder_path}")
        return pdf_files
        
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                # Kiểm tra file có thể đọc được không
                if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                    pdf_files.append(full_path)
                else:
                    print(f"⚠️ Không thể đọc file: {full_path}")
    return pdf_files

pdf_files = get_all_pdf_files(input_folder)
print(f"🔍 Tìm thấy {len(pdf_files)} file PDF trong folder và các thư mục con: {input_folder}")

# In ra danh sách các thư mục con được tìm thấy
subfolders = []
for root, dirs, files in os.walk(input_folder):
    for dir_name in dirs:
        subfolders.append(os.path.join(root, dir_name))
        
if subfolders:
    print("📁 Các thư mục con được tìm thấy:")
    for subfolder in subfolders:
        subfolder_pdfs = [f for f in os.listdir(subfolder) if f.lower().endswith('.pdf')]
        print(f"   - {subfolder}: {len(subfolder_pdfs)} file PDF")

processed_count = 0
skipped_count = 0
error_count = 0

for i, pdf_path in enumerate(pdf_files, 1):
    pdf_file = os.path.basename(pdf_path)  # Get just the filename for display
    relative_path = os.path.relpath(pdf_path, input_folder)  # Get relative path from documents folder
    
    print(f"\n{'='*80}")
    print(f"📄 [{i}/{len(pdf_files)}] Xử lý: {relative_path}")
    print(f"{'='*80}")
    
    try:
        if is_pdf_embedded(pdf_path):
            print(f"📌 File {pdf_file} đã được embedding trước đó, bỏ qua.")
            skipped_count += 1
            continue
        
        print(f"🚀 Bắt đầu xử lý: {pdf_path}")
        raw_text = ocr_pdf_to_text(pdf_path, output_dir)
        if raw_text and raw_text.strip():
            print(f"🧹 Làm sạch text cho {pdf_file}...")
            cleaned_text = clean_text(raw_text, pdf_path, output_dir)
            if cleaned_text and cleaned_text.strip():
                print(f"✂️ Chia text thành chunks cho {pdf_file}...")
                chunks = split_text_to_chunks_vi_tokenized_with_section(cleaned_text)
                if chunks:
                    print(f"📝 Đã tạo {len(chunks)} chunks cho {pdf_file}")
                    embeddings = create_embeddings(chunks)
                    if embeddings is not None:
                        pickle_path, faiss_path = save_embeddings(chunks, embeddings, pdf_path, output_dir)
                        print(f"\n🎉 HOÀN THÀNH embedding cho {pdf_file}!")
                        print(f"📊 Thống kê: - Số chunks: {len(chunks)} - Kích thước embedding: {embeddings.shape}")
                        print(f"✅ Đường dẫn embeddings: {pickle_path}")
                        print(f"✅ Đường dẫn FAISS index: {faiss_path}")
                        processed_count += 1
                    else:
                        print(f"❌ Không thể tạo embeddings cho {pdf_file}")
                        error_count += 1
                else:
                    print(f"❌ Không thể tạo chunks từ text của {pdf_file}")
                    error_count += 1
            else:
                print(f"❌ Text rỗng sau khi làm sạch cho {pdf_file}")
                error_count += 1
        else:
            print(f"❌ Không thể OCR text từ {pdf_file}")
            error_count += 1
            
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {pdf_file}: {str(e)}")
        error_count += 1
        continue

# Tổng kết quá trình xử lý
print(f"\n{'='*80}")
print("📊 TỔNG KẾT QUÁ TRÌNH XỬ LÝ")
print(f"{'='*80}")
print(f"📄 Tổng số file PDF tìm thấy: {len(pdf_files)}")
print(f"✅ Đã xử lý thành công: {processed_count}")
print(f"📌 Đã bỏ qua (đã embedding): {skipped_count}")
print(f"❌ Lỗi xử lý: {error_count}")
print(f"📁 Thư mục kết quả: {output_dir}")
print(f"🔗 FAISS index tổng hợp: {all_faiss_path}")
print(f"📦 Pickle tổng hợp: {all_pickle_path}")

if processed_count > 0:
    print(f"\n🎉 Đã cập nhật knowledge base với {processed_count} tài liệu mới!")
else:
    print(f"\n📋 Không có tài liệu mới nào được thêm vào knowledge base.")
    
print(f"{'='*80}")