import torch
import pandas as pd
import glob
import re
import os

# 1. Đường dẫn tới thư mục chứa các file .pt của bạn
# Nếu chạy trong thư mục src, dùng "../checkpoints" để ra ngoài tìm folder checkpoints
checkpoint_dir = "../checkpoints" 
output_csv = "training_log_extracted.csv"

history = []

# 2. Tìm tất cả các file .pt và đếm tổng số
files = glob.glob(os.path.join(checkpoint_dir, "ctc_model_epoch*.pt"))
total_files = len(files)

if total_files == 0:
    print(f"❌ Không tìm thấy file .pt nào trong thư mục: {os.path.abspath(checkpoint_dir)}")
else:
    print(f"🚀 Tìm thấy {total_files} file checkpoint. Bắt đầu quét...")
    print("-" * 50)

    for i, f in enumerate(files):
        try:
            # Hiển thị tiến trình đang chạy
            filename = os.path.basename(f)
            print(f"🔄 [{i+1}/{total_files}] Đang đọc: {filename}...", end='\r')
            
            # Trích xuất số epoch từ tên file
            match = re.search(r'epoch(\d+)', filename)
            epoch_num = int(match.group(1)) if match else i
            
            # Load file (weights_only=True giúp chạy nhanh hơn nếu bản Torch hỗ trợ)
            checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            
            # Lấy giá trị loss (Thử các key phổ biến nếu 'loss' không có)
            loss_value = checkpoint.get('loss') or checkpoint.get('train_loss') or 0.0
            
            history.append({
                "Epoch": epoch_num,
                "Train_Loss": loss_value,
                "Val_Loss": "N/A"
            })
            
        except Exception as e:
            print(f"\n❌ Lỗi ở file {f}: {e}")

    # 3. Sắp xếp và lưu file
    if history:
        print(f"\n\n📊 Đang tổng hợp dữ liệu và lưu CSV...")
        df = pd.DataFrame(history)
        df = df.sort_values(by="Epoch")
        df.to_csv(output_csv, index=False)
        print("-" * 50)
        print(f"🏁 XONG! Đã tạo file: {os.path.abspath(output_csv)}")
    else:
        print("\n⚠️ Không có dữ liệu nào được trích xuất.")