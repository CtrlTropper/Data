# %% [markdown]
# # Khai báo thư viện

 # Thêm sentence-transformers cho multilingual_e5_large

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import time
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import re
import json
from pathlib import Path  # tiện hơn os.path
from underthesea import sent_tokenize
from transformers import AutoTokenizer
import faiss
import shutil
import torch

print(f"Số lượng GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(
        f"  Tổng VRAM: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024**2:.2f} GB"
    )

device = torch.device("cuda:0")  # Chỉ định GPU 0 (Tesla T4)
print(f"\nĐang sử dụng: {torch.cuda.get_device_name(device)}")

# %%
# Thay đổi: Sử dụng model multilingual_e5_large từ HuggingFace (hỗ trợ đa ngôn ngữ: TV, Anh, Pháp, Trung, v.v.)
# Model này cho phép embedding tài liệu đa ngôn ngữ mà không cần dịch, và matching cross-language với query TV.
# Nếu chưa tải, SentenceTransformer sẽ tự động tải từ HF repo 'intfloat/multilingual-e5-large'.
model_path = "intfloat/multilingual-e5-large"
try:
    model = SentenceTransformer(model_path, device='cuda:0')  # Load với GPU để tăng tốc
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✅ Loaded multilingual-e5-large successfully. Supports cross-lingual semantic similarity.")
except Exception as e:
    print(f"❌ Error loading model: {e}. Falling back to CPU or reinstall sentence-transformers.")

# %% [markdown]
# # Tạo embeddings từ PDF

# %%
# Đường dẫn file PDF bạn muốn xử lý
pdf_path = "../data/file/CTKM_01.pdf"
output_dir = "./results"

all_faiss_path = os.path.join(output_dir, "all_faiss.index")
all_pickle_path = os.path.join(output_dir, "all_embeddings.pkl")

# %% [markdown]
# ## Bước 1: Check xem file pdf đó đã được embedding chưa

# %%
def is_pdf_embedded(path):
    """
    Kiểm tra xem file PDF đã được embedding hay chưa,
    dựa vào all_embeddings.pkl (danh sách các file đã xử lý).
    """
    if not os.path.exists(all_pickle_path):
        return False  # Chưa có dữ liệu chung => chắc chắn chưa nhúng gì

    pdf_name = os.path.splitext(os.path.basename(path))[0]

    with open(all_pickle_path, 'rb') as f:
        all_data = pickle.load(f)

    existing_pdf_names = {entry['pdf_name'] for entry in all_data}

    return pdf_name in existing_pdf_names

# %%
if is_pdf_embedded(pdf_path):
    print("📌 PDF này đã được embedding trước đó.")
else:
    print("🔄 PDF này chưa được embedding.")

# %% [markdown]
# ## Bước 2: OCR PDF

# %%
def preprocess_image(img):
    """
    Tiền xử lý ảnh để cải thiện OCR
    """
    if img.mode != 'L':
        img = img.convert('L')

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    img = img.filter(ImageFilter.SHARPEN)

    return img

# %%
def ocr_pdf_to_text(pdf_path, output_dir):
    """
    OCR file PDF thành text
    """
    try:
        print(f"📖 Đang OCR file: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        full_text = ""
        ocr_config = r'--oem 3 --psm 6 -l vie'  # Giữ config tiếng Việt, nhưng model embedding sẽ xử lý đa ngôn ngữ sau

        for page_num in range(total_pages):
            print(f"🔄 Xử lý trang {page_num + 1}/{total_pages}...")

            page = doc.load_page(page_num)
            matrix = fitz.Matrix(2.5, 2.5)
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            img = preprocess_image(img)

            try:
                page_text = pytesseract.image_to_string(img, config=ocr_config)
                full_text += page_text.strip()

            except Exception as e:
                print(f"   ❌ Lỗi OCR trang {page_num + 1}: {e}")

        doc.close()
        print(f"✅ Hoàn thành OCR {total_pages} trang")

        # 🧠 Tạo tên file theo tên file PDF
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_ocr.txt")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"📄 Kết quả đã lưu vào: {output_path}")

        return full_text

    except Exception as e:
        print(f"❌ Lỗi OCR: {e}")
        return None

# %%
# Bước 1: OCR PDF
raw_text = ocr_pdf_to_text(pdf_path, output_dir)
if not raw_text:
    print("❌ Không thể OCR file PDF. Vui lòng kiểm tra lại.")
    exit(1)

# ## Bước 3: Làm sạch text

def clean_text(text, pdf_path, output_dir):
    """
    Làm sạch một đoạn văn bản OCR (string)
    """
    # Loại ký tự không mong muốn (giữ lại tiếng Việt, toán học, đơn vị)
    text = re.sub(r'[^\w\s.,;:()\[\]?!\"\'\-–—…°%‰≥≤→←≠=+/*<>\n\r]', '', text)

    # Xử lý lỗi xuống dòng giữa từ hoặc giữa câu
    text = re.sub(r'-\n', '', text)             # nối từ bị gạch nối xuống dòng
    text = re.sub(r'\n(?=\w)', ' ', text)       # dòng xuống không hợp lý → nối câu

    # Dấu chấm lặp vô nghĩa → ba chấm
    text = re.sub(r'\.{3,}', '...', text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\n\s*\n', '\n\n', text)   # giữ ngắt đoạn
    text = re.sub(r'[ \t]+', ' ', text)       # nhiều khoảng trắng → 1 dấu cách
    text = re.sub(r' *\n *', '\n', text)      # bỏ khoảng trắng đầu/cuối dòng

    # Lưu file
    clean_text = text.strip()
    
    # 🧠 Tạo tên file theo tên file PDF
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_clean.txt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"📄 Kết quả đã lưu vào: {output_path}")
    
    return clean_text


# Bước 2: Làm sạch text
print("🧹 Làm sạch text...")
cleaned_text = clean_text(raw_text, pdf_path, output_dir)

# ## Bước 4: Chia thành chunks

def split_sections(text):
    """
    Tách text thành các phần theo tiêu đề kiểu I., 1., a)
    """
    sections = re.split(r'\n(?=(?:[IVXLCDM]+\.)|(?:\d+\.)|(?:[a-z]\)))', text)
    return [s.strip() for s in sections if s.strip()]

def split_text_to_chunks_vi_tokenized_with_section(text, chunk_size=512, overlap=50):
    """
    Chia văn bản tiếng Việt thành các chunk dựa trên số token,
    giữ nguyên cấu trúc section và câu.
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

                # Overlap bằng token
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

# Bước 3: Chia thành chunks
print("✂️ Chia text thành chunks...")
chunks = split_text_to_chunks_vi_tokenized_with_section(cleaned_text)
print(f"📝 Đã tạo {len(chunks)} chunks")

# ## Bước 5: Tạo embeddings

def create_embeddings(chunks):
    """
    Tạo embeddings cho các text chunks
    """
    try:
        print(f"🔄 Tạo embeddings cho {len(chunks)} chunks...")
        # Thay đổi: Encode với multilingual_e5_large, thêm batch_size=32 để hiệu quả với GPU
        # Model hỗ trợ đa ngôn ngữ, không cần preprocess thêm cho cross-lang matching
        embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)  # Normalize để cosine sim chính xác hơn
        print(f"✅ Hoàn thành tạo embeddings")
        return embeddings

    except Exception as e:
        print(f"❌ Lỗi tạo embeddings: {e}")
        return None

# Bước 4: Tạo embeddings
embeddings = create_embeddings(chunks)
if embeddings is None:
    print("❌ Không thể tạo embeddings. Vui lòng kiểm tra lại.")
    exit(1)

# ## Bước 6: Lưu embeddings

def save_embeddings(chunks, embeddings, pdf_path, output_dir):
    """
    Lưu embeddings và chunks vào file
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.join(output_dir, pdf_name), exist_ok=True)

    # Lưu dữ liệu
    data = {
        'pdf_name': pdf_name,
        'chunks': chunks,
        'embeddings': embeddings,
        'created_at': datetime.now().isoformat()
    }

    # Lưu embeddings (pickle)
    pickle_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_embeddings.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    # Lưu chunks (text file)
    chunks_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_chunks.txt")
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

    # Lưu thông tin embeddings (text file)
    embedding_info_path = os.path.join(
        output_dir, pdf_name, f"{pdf_name}_embedding_info.txt")
    with open(embedding_info_path, 'w', encoding='utf-8') as f:
        f.write(f"THÔNG TIN EMBEDDINGS: {pdf_name}.pdf\n")
        f.write(f"Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"📊 THỐNG KÊ:\n")
        f.write(f"- Tổng số chunks: {len(chunks)}\n")
        f.write(f"- Kích thước embeddings: {embeddings.shape}\n")
        f.write(f"- Kiểu dữ liệu: {embeddings.dtype}\n")
        f.write(
            f"- Kích thước mỗi vector: {embeddings.shape[1]} dimensions\n\n")

        f.write(f"📝 PREVIEW EMBEDDINGS (5 chunks đầu):\n")
        f.write("-" * 50 + "\n")

        for i in range(min(5, len(chunks))):
            f.write(f"\nCHUNK {i+1}:\n")
            f.write(f"Text: {chunks[i][:100]}...\n")
            f.write(
                f"Embedding vector (10 giá trị đầu): {embeddings[i][:10].tolist()}\n")
            f.write(f"Vector norm: {np.linalg.norm(embeddings[i]):.4f}\n")
            f.write("-" * 30 + "\n")
    
    # 4️⃣ Lưu FAISS index
    index_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_faiss.index")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    # --- 🔁 Cập nhật FAISS chung ---
    all_faiss_path = os.path.join(output_dir, "all_faiss.index")
    if os.path.exists(all_faiss_path):
        index_all = faiss.read_index(all_faiss_path)
    else:
        index_all = faiss.IndexFlatL2(dim)

    index_all.add(embeddings.astype(np.float32))
    faiss.write_index(index_all, all_faiss_path)

    # --- 🔁 Cập nhật pickle chung ---
    all_pickle_path = os.path.join(output_dir, "all_embeddings.pkl")
    if os.path.exists(all_pickle_path):
        with open(all_pickle_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        all_data = []

    all_data.append(data)

    with open(all_pickle_path, 'wb') as f:
        pickle.dump(all_data, f)

    print(f"💾 Đã lưu embeddings: {pickle_path}")
    print(f"📄 Đã lưu chunks: {chunks_path}")
    print(f"📊 Đã lưu thông tin embeddings: {embedding_info_path}")
    print(f"📌 Đã lưu FAISS index: {index_path}")
    print(f"🔁 Cập nhật FAISS chung: {all_faiss_path}")
    print(f"📦 Cập nhật pickle chung: {all_pickle_path}")

    return pickle_path, index_path

# Bước 5: Lưu embeddings
pickle_path, faiss_path = save_embeddings(chunks, embeddings, pdf_path, output_dir)

print(f"\n🎉 HOÀN THÀNH!")
print(f"📊 Thống kê:")
print(f"   - Số chunks: {len(chunks)}")
print(f"   - Kích thước embedding: {embeddings.shape}")
print(f"   - File embeddings: {pickle_path}")
print(
    f"   - File chunks: {pickle_path.replace('_embeddings.pkl', '_chunks.txt')}")
print(
    f"   - File thông tin: {pickle_path.replace('_embeddings.pkl', '_embedding_info.txt')}")

# Lưu lại đường dẫn file .pkl để sử dụng sau
print(f"✅ Đường dẫn embeddings: {pickle_path}")
print(f"✅ Đường dẫn FAISS index: {faiss_path}")

# ## Xóa pdf

# Xóa thư mục chứa embeddings của file PDF
def delete_pdf_folder(pdf_path, output_dir):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    folder_path = os.path.join(output_dir, pdf_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"🗑️ Đã xoá thư mục: {folder_path}")
    else:
        print(f"⚠️ Thư mục không tồn tại: {folder_path}")

# Xóa khỏi all_embeddings.pkl
def remove_from_all_embeddings(pdf_path, all_pickle_path):
    if not os.path.exists(all_pickle_path):
        print("⚠️ all_embeddings.pkl không tồn tại.")
        return

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    with open(all_pickle_path, 'rb') as f:
        all_data = pickle.load(f)

    new_data = [entry for entry in all_data if entry['pdf_name'] != pdf_name]

    if len(new_data) == len(all_data):
        print(f"❌ PDF '{pdf_name}' không có trong all_embeddings.pkl.")
    else:
        with open(all_pickle_path, 'wb') as f:
            pickle.dump(new_data, f)
        print(f"✅ Đã xoá '{pdf_name}' khỏi all_embeddings.pkl.")


# Xóa khỏi all_faiss.index
def rebuild_faiss_from_pickle(all_pickle_path, all_faiss_path):
    if not os.path.exists(all_pickle_path):
        print("⚠️ all_embeddings.pkl không tồn tại.")
        return

    with open(all_pickle_path, 'rb') as f:
        all_data = pickle.load(f)

    if not all_data:
        print("⚠️ Không còn dữ liệu nào trong all_embeddings.pkl.")
        if os.path.exists(all_faiss_path):
            os.remove(all_faiss_path)
            print("🗑️ Đã xoá all_faiss.index rỗng.")
        return

    # Gom tất cả vectors lại
    all_vectors = np.vstack([entry['embeddings'] for entry in all_data]).astype(np.float32)
    dim = all_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_vectors)
    faiss.write_index(index, all_faiss_path)
    print(f"🔄 Đã xây lại all_faiss.index với {all_vectors.shape[0]} vectors.")

def delete_pdf_folder_and_update(pdf_path, output_dir):
    all_pickle_path = os.path.join(output_dir, "all_embeddings.pkl")
    all_faiss_path = os.path.join(output_dir, "all_faiss.index")
    
    delete_pdf_folder(pdf_path, output_dir)
    remove_from_all_embeddings(pdf_path, all_pickle_path)
    rebuild_faiss_from_pickle(all_pickle_path, all_faiss_path)

delete_pdf_folder_and_update(pdf_path, output_dir)

# # Demo tìm kiếm với câu hỏi bất kỳ
# Thử tìm kiếm với câu hỏi bất kỳ
query = "Trụ sở của trường nằm ở đâu?"

## Dùng Pickle

## Bước 1: Load embedding

# %%
def load_embeddings(embeddings_path):
    """
    Tải embeddings từ file
    """
    try:
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)

        print(f"📂 Đã tải embeddings từ: {embeddings_path}")
        return data

    except Exception as e:
        print(f"❌ Lỗi tải embeddings: {e}")
        return None

# %%
# Tải lại dữ liệu embeddings đã tạo
data = load_embeddings(pickle_path)

# %% [markdown]
## Bước 2: Tìm kiếm

# %%
def find_similar_chunks(query, embeddings_data, top_k=3):
    """
    Tìm các chunk tương tự với query
    """
    try:
        # Thay đổi: Embed query với multilingual_e5_large (query TV matching chunks đa ngôn ngữ)
        query_embedding = model.encode([query], normalize_embeddings=True)

        # Tính cosine similarity
        similarities = np.dot(
            embeddings_data['embeddings'], query_embedding.T).flatten()

        # Lấy top_k kết quả
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'chunk': embeddings_data['chunks'][idx],
                'similarity': similarities[idx],
                'index': idx
            })

        return results

    except Exception as e:
        print(f"❌ Lỗi tìm kiếm: {e}")
        return []

# %%
results = find_similar_chunks(query, data, top_k=3)

print(f"\n🔍 DEMO TÌM KIẾM")
print(f"📚 PDF: {data['pdf_name']}")
print(f"📝 Số chunks: {len(data['chunks'])}")
print("=" * 50)

# In kết quả
for i, result in enumerate(results, 1):
    print(f"\n{i}. 🔍 Độ tương đồng: {result['similarity']:.4f}")
    print(f"📄 Nội dung: {result['chunk'][:300]}...")

## Dùng FAISS
## Bước 1: Load FAISS index
def load_faiss_index(index_path):
    print(f"📂 Đang tải FAISS index từ: {index_path}")
    return faiss.read_index(index_path)

index = load_faiss_index(faiss_path)

## Bước 2: Encode câu hỏi thành vector

def encode_query(query):
    # Thay đổi: Embed query với normalize để matching tốt hơn với chunks đa ngôn ngữ
    return model.encode([query], normalize_embeddings=True)  # trả về numpy array (1, dim)

query_vector = encode_query(query)

## Bước 3: Truy vấn FAISS
def search_faiss(index, query_vector, top_k=5):
    D, I = index.search(query_vector.astype('float32'), top_k)
    return I[0], D[0]  # indices, distances

top_k = 3
indices, distances = search_faiss(index, query_vector, top_k)

print("📌 Kết quả truy vấn:")
for i, (idx, dist) in enumerate(zip(indices, distances), 1):
    print(f"\n🔹 Kết quả {i}: (score: {dist:.4f})")
    print(chunks[idx][:500])  # in tối đa 500 ký tự