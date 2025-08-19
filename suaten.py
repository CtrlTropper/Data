import os

# Bảng chuyển đổi để loại bỏ dấu tiếng Việt
VI_CHARS = 'àáảãạâấầẩẫậăắằẳẵặèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
NO_CHARS = 'aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd'
TRANS_TABLE = str.maketrans(VI_CHARS + VI_CHARS.upper(), NO_CHARS + NO_CHARS.upper())

def remove_accents(s):
    return s.translate(TRANS_TABLE)

def normalize_filename(filename):
    base, ext = os.path.splitext(filename)
    base = base.replace(" ", "_")
    base = remove_accents(base)
    # Tối ưu để tránh "_-" (nếu có khoảng trắng trước "-")
    base = base.replace("_-", "-")
    return base + ext

def rename_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print("Thư mục không tồn tại!")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            new_filename = normalize_filename(filename)
            new_file_path = os.path.join(folder_path, new_filename)
            if new_filename != filename:
                os.rename(file_path, new_file_path)
                print(f"Đổi tên: {filename} --> {new_filename}")
            else:
                print(f"Giữ nguyên: {filename}")

if __name__ == "__main__":
    folder = input("Nhập đường dẫn thư mục chứa file: ")
    rename_files_in_folder(folder)
# D:/DoAnTotNghiep/Data/LuatVietNam