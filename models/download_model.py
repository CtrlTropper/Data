# from huggingface_hub import snapshot_download

# Tên mô hình trên Hugging Face
# model_name = "vilm/vinallama-2.7b-chat"

# # Tải mô hình về thư mục cục bộ
# snapshot_download(
#     repo_id=model_name,
#     local_dir="./vinallama-2.7b-chat",  # Thư mục lưu mô hình
#     local_dir_use_symlinks=False,       # Không sử dụng liên kết tượng trưng
#     cache_dir="./cache"                 # Thư mục tạm để lưu cache
# )

# print(f"Mô hình {model_name} đã được tải về thư mục ./vinallama-2.7b-chat")
# --------------------------------------------------------------------------------
# from transformers import AutoTokenizer, AutoModel
# import torch
# import os

# # Định nghĩa tên mô hình
# model_name = "intfloat/multilingual-e5-large"

# # Thư mục để lưu mô hình
# save_directory = "./multilingual_e5_large"

# # Tạo thư mục nếu chưa tồn tại
# os.makedirs(save_directory, exist_ok=True)

# # Tải tokenizer
# print("Đang tải tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Tải mô hình
# print("Đang tải mô hình...")
# model = AutoModel.from_pretrained(model_name)

# # Lưu tokenizer và mô hình vào thư mục
# print(f"Đang lưu vào {save_directory}...")
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)

# print("Tải và lưu mô hình thành công!")

# # Ví dụ sử dụng mô hình để tạo embedding
# def get_embedding(text):
#     # Tokenize văn bản
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
#     # Tạo embedding
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Lấy embedding từ [CLS] token hoặc mean pooling
#         embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    
#     return embedding

# # Thử nghiệm
# sample_text = "Xin chào, đây là một ví dụ tiếng Việt và English mixed."
# embedding = get_embedding(sample_text)
# print("Embedding shape:", embedding.shape)
# print("Embedding sample:", embedding[:5])
# ----------------------------------------------------------------------------------------------------------
from huggingface_hub import snapshot_download

# Tên mô hình trên Hugging Face
model_name = "openai/gpt-oss-20b"

# Tải mô hình về thư mục cục bộ
snapshot_download(
    repo_id=model_name,
    local_dir="./gpt-oss-20b",      # Thư mục lưu mô hình
    local_dir_use_symlinks=False,   # Không sử dụng liên kết tượng trưng
    cache_dir="./cache"             # Thư mục tạm để lưu cache
)

print(f"Mô hình {model_name} đã được tải về thư mục ./gpt-oss-20b")