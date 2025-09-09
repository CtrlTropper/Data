#!/usr/bin/env python3
"""
Script để cài đặt môi trường cần thiết cho embedding.py
Setup script for installing required packages for embedding.py
"""

import subprocess
import sys
import os

def install_package(package_name, pip_name=None):
    """Cài đặt package với tên pip khác nếu cần"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        __import__(package_name)
        print(f"✅ {package_name} đã được cài đặt")
        return True
    except ImportError:
        print(f"🔄 Đang cài đặt {pip_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", pip_name, 
                "--break-system-packages"
            ])
            print(f"✅ Đã cài đặt thành công {pip_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Không thể cài đặt {pip_name}")
            return False

def main():
    print("🚀 Bắt đầu cài đặt môi trường cho embedding.py")
    print("=" * 60)
    
    # Danh sách các package cần thiết
    packages = [
        ("fitz", "PyMuPDF"),
        ("pytesseract", "pytesseract"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("sentence_transformers", "sentence-transformers"),
        ("transformers", "transformers"),
        ("faiss", "faiss-cpu"),
        ("underthesea", "underthesea"),
        ("torch", "torch"),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for import_name, pip_name in packages:
        if install_package(import_name, pip_name):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 KẾT QUẢ: {success_count}/{total_count} packages đã được cài đặt thành công")
    
    if success_count == total_count:
        print("🎉 Môi trường đã sẵn sàng! Bạn có thể chạy embeding.py")
    else:
        print("⚠️ Một số packages chưa được cài đặt. Vui lòng cài đặt thủ công:")
        print("   pip install PyMuPDF pytesseract pillow sentence-transformers transformers faiss-cpu underthesea torch numpy --break-system-packages")

if __name__ == "__main__":
    main()