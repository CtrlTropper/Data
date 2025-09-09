# %pip install langchain langchain-community langchain-openai
# !pip install underthesea
import io
from datetime import datetime
import pickle
import os
import re
import argparse
from typing import Optional

# Model/tokenizer sẽ được load lười khi cần
model = None  # type: ignore[var-annotated]
tokenizer = None  # type: ignore[var-annotated]

def ensure_model_and_tokenizer_loaded():
    global model, tokenizer
    if model is None or tokenizer is None:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
        model_path = os.environ.get("EMBED_MODEL_PATH", "intfloat/multilingual-e5-large")
        model = SentenceTransformer(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

# Đường dẫn input folder chứa tài liệu (xử lý đệ quy tất cả thư mục con)
input_folder = os.environ.get("DOCUMENTS_DIR", "/workspace/documents")
output_dir = os.environ.get("EMBED_RESULTS_DIR", "./results")
os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục output nếu chưa có

# Phần mở rộng tài liệu được hỗ trợ
SUPPORTED_EXTS = {".pdf", ".txt", ".docx"}

all_faiss_path = os.path.join(output_dir, "all_faiss.index")
all_pickle_path = os.path.join(output_dir, "all_embeddings.pkl")

def is_document_embedded(path):
    """
    Kiểm tra xem tài liệu đã được embedding hay chưa dựa trên file pickle chung.
    Vẫn tương thích với dữ liệu cũ sử dụng key 'pdf_name'.
    """
    if not os.path.exists(all_pickle_path):
        return False
    base_name = os.path.splitext(os.path.basename(path))[0]
    with open(all_pickle_path, 'rb') as f:
        all_data = pickle.load(f)
    existing_names = set()
    for entry in all_data:
        if isinstance(entry, dict):
            if 'pdf_name' in entry:
                existing_names.add(entry['pdf_name'])
            if 'doc_name' in entry:  # phòng trường hợp về sau đổi key
                existing_names.add(entry['doc_name'])
    return base_name in existing_names

def preprocess_image(img):
    """
    Tiền xử lý ảnh để cải thiện OCR.
    """
    # Import lười để hỗ trợ --dry-run không cần pillow
    from PIL import ImageEnhance, ImageFilter
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
        # Import lười để hỗ trợ --dry-run
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

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

def read_txt_to_text(txt_path):
    """
    Đọc nội dung từ file .txt (thử nhiều encoding phổ biến).
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(txt_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Lỗi đọc TXT {txt_path}: {e}")
        return None

def read_docx_to_text(docx_path):
    """
    Trích xuất text từ file .docx.
    """
    try:
        # Import lười để hỗ trợ --dry-run
        import docx2txt
        text = docx2txt.process(docx_path) or ""
        return text
    except Exception as e:
        print(f"❌ Lỗi đọc DOCX {docx_path}: {e}")
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
    # Import lười để hỗ trợ --dry-run
    from underthesea import sent_tokenize
    ensure_model_and_tokenizer_loaded()
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
        ensure_model_and_tokenizer_loaded()
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
        'pdf_name': pdf_name,  # giữ key cũ để tương thích
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
    import numpy as np
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
    import faiss
    index_path = os.path.join(pdf_folder, f"{pdf_name}_faiss.index")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    # Cập nhật FAISS chung
    import faiss
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

def discover_documents(root_dir, exts):
    """
    Duyệt đệ quy để lấy tất cả tài liệu hợp lệ trong thư mục.
    """
    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in exts:
                yield os.path.join(current_root, file_name)


def extract_text_any(file_path, output_dir):
    """
    Trích xuất text từ tài liệu theo phần mở rộng.
    PDF: sử dụng OCR hiện tại.
    TXT/DOCX: đọc trực tiếp nội dung.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return ocr_pdf_to_text(file_path, output_dir)
    if ext == ".txt":
        return read_txt_to_text(file_path)
    if ext == ".docx":
        return read_docx_to_text(file_path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Embed documents recursively")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ liệt kê tài liệu, không xử lý")
    args = parser.parse_args()

    # Quy trình chính: Embedding tất cả tài liệu trong input_folder và các thư mục con
    all_docs = list(discover_documents(input_folder, SUPPORTED_EXTS))
    print(f"🔍 Tìm thấy {len(all_docs)} tài liệu trong thư mục: {input_folder}")

    if args.dry_run:
        for path in all_docs[:50]:  # giới hạn in để gọn log
            print(f" - {path}")
        if len(all_docs) > 50:
            print(f" ... và {len(all_docs) - 50} tài liệu khác")
        return

    for doc_path in all_docs:
        doc_name = os.path.basename(doc_path)
        if is_document_embedded(doc_path):
            print(f"📌 File {doc_name} đã được embedding trước đó, bỏ qua.")
            continue

        raw_text = extract_text_any(doc_path, output_dir)
        if raw_text:
            print(f"🧹 Làm sạch text cho {doc_name}...")
            cleaned_text = clean_text(raw_text, doc_path, output_dir)
            print(f"✂️ Chia text thành chunks cho {doc_name}...")
            chunks = split_text_to_chunks_vi_tokenized_with_section(cleaned_text)
            print(f"📝 Đã tạo {len(chunks)} chunks cho {doc_name}")
            embeddings = create_embeddings(chunks)
            if embeddings is not None:
                pickle_path, faiss_path = save_embeddings(chunks, embeddings, doc_path, output_dir)
                print(f"\n🎉 HOÀN THÀNH embedding cho {doc_name}!")
                print(f"📊 Thống kê: - Số chunks: {len(chunks)} - Kích thước embedding: {embeddings.shape}")
                print(f"✅ Đường dẫn embeddings: {pickle_path}")
                print(f"✅ Đường dẫn FAISS index: {faiss_path}")

    print("\n✅ Đã xử lý tất cả tài liệu trong thư mục. Knowledge base đã cập nhật!")


if __name__ == "__main__":
    main()