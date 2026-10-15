import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# --- 1. THIẾT LẬP ĐƯỜNG DẪN (PATH SETUP) ---
# Tự động lấy thư mục gốc của project (nơi bạn chạy "python train.py")
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import từ các file .py trong 'src'
# (Đảm bảo bạn đã sửa lỗi import lồng nhau, ví dụ: 'from .tokenizer import Tokenizer' 
# bên trong file dataset.py)
try:
    from src.data.dataset import ASRDataset, custom_collate_fn 
    from src.data.tokenizer import Tokenizer
    from src.models.asr_model import ASRModel
except ImportError as e:
    print(f"LỖI: Không thể import từ thư mục 'src'.")
    print(f"Đảm bảo bạn đã có file __init__.py trong 'src', 'src/data', 'src/models'.")
    print(f"Lỗi gốc: {e}")
    sys.exit(1)

# --- 2. HÀM HUẤN LUYỆN (TRAIN FUNCTION) ---
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """
    Hàm logic để chạy 1 epoch huấn luyện, có hỗ trợ AMP (GradScaler).
    """
    model.train()
    total_loss = 0.0
    
    # Bọc dataloader bằng tqdm để xem tiến độ
    pbar = tqdm(dataloader, desc=f"Training Epoch")
    for batch in pbar:
        if batch is None:
            continue
            
        inputs = batch["input_features"].to(device) # (B, 1, 80, T)
        labels = batch["labels"].to(device)
        input_lens = batch["input_lengths"].to(device) # Độ dài T ban đầu
        label_lens = batch["label_lengths"].to(device)

        optimizer.zero_grad()
        
        # Dùng autocast (AMP)
        # Sửa: Thêm device_type='cuda'
        with autocast():
            log_probs = model(inputs) # (T', B, Vocab)
            
            # Lấy độ dài T' (sau khi qua CNN) từ model
            output_lens = model.get_output_lengths(input_lens)

            # Tính loss
            loss = criterion(log_probs, labels, output_lens, label_lens)
        
        # Dùng GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}") # Cập nhật loss lên thanh tqdm
        
    return total_loss / len(dataloader)

# --- 3. HÀM CHÍNH (MAIN FUNCTION) ---
def main():
    # === Cấu hình (Config) ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Thiết bị sử dụng: {device}")
    if not torch.cuda.is_available():
        print("⚠️ CẢNH BÁO: Không tìm thấy GPU. Đang chạy trên CPU (rất chậm).")

    # Đường dẫn (lấy từ project_root đã định nghĩa ở trên)
    MANIFEST_PATH = os.path.join(project_root, "data", "manifests", "train_all.csv")
    VOCAB_PATH = os.path.join(project_root, "data", "tokenizer_vocab.json")
    CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
    RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ctc_model_last.pt")

    # === Tối ưu cho GPU Local (PC của bạn) ===
    # Bạn có thể tăng BATCH_SIZE nếu GPU của bạn mạnh
    BATCH_SIZE = 16  # (Tăng từ 16 lên 32)
    # Tăng NUM_WORKERS để tận dụng CPU đa lõi khi load data
    NUM_WORKERS = 4  # (Tăng từ 2 lên 4. Chỉnh về 0 nếu bị lỗi trên Windows)
    
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 90

    # Cấu hình Model (phải khớp với checkpoint)
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    N_MELS = 80

    print("✅ Cấu hình đã được tải.")

    # === 1. Tải Tokenizer ===
    print(f"Đang tải Tokenizer từ: {VOCAB_PATH}")
    tokenizer = Tokenizer(vocab_path=VOCAB_PATH)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}, PAD_ID=0, BLANK_ID={tokenizer.blank_id}")

    # === 2. Tải Dataset & DataLoader ===
    print(f"Đang tải Dataset từ: {MANIFEST_PATH}")
    dataset = ASRDataset(
        manifest_path=MANIFEST_PATH, 
        vocab_path=VOCAB_PATH,
        n_mels=N_MELS,
        use_spec_augment=True # Bật Augmentation khi train
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True # Rất tốt cho GPU
    )
    print(f"✅ Tải Dataset và DataLoader thành công với {len(dataset)} mẫu.")

    # === 3. Khởi tạo Model, Loss, Optimizer ===
    model = ASRModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        n_mels=N_MELS
    ).to(device)

    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True) 
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # Sửa: Thêm device_type='cuda'
    scaler = GradScaler()
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = 0

    print("✅ Khởi tạo Model, Loss, Optimizer thành công.")

    # === 4. Tải Checkpoint (Resume) ===
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Đang tải checkpoint từ: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"]) 
            start_epoch = checkpoint["epoch"] + 1
            print(f"🔄 Resume training từ epoch {start_epoch}")
        except KeyError:
            print("Checkpoint cũ không tương thích, có thể thiếu 'scaler_state'. Bắt đầu lại từ đầu.")
            start_epoch = 0 # Reset nếu checkpoint lỗi
    else:
        print("Bắt đầu huấn luyện từ đầu.")

    # === 5. Vòng lặp Huấn luyện ===
    print("🚀 Bắt đầu huấn luyện...")
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            
            # Chạy 1 epoch
            avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, scaler, device)
            
            print(f"✅ Epoch {epoch+1}/{NUM_EPOCHS} | Avg Loss = {avg_loss:.4f}")

            # Lưu checkpoint
            current_ckpt_path = os.path.join(CHECKPOINT_DIR, f"ctc_model_epoch{epoch+1}.pt")
            last_ckpt_path = os.path.join(CHECKPOINT_DIR, "ctc_model_last.pt")
            
            save_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(), 
                "loss": avg_loss,
            }
            
            torch.save(save_dict, current_ckpt_path)
            torch.save(save_dict, last_ckpt_path) # Ghi đè file _last

            print(f"💾 Đã lưu checkpoint: {current_ckpt_path}")

        print("🎉 Huấn luyện hoàn tất!")

    except KeyboardInterrupt:
        print("\n⏹️ Tạm dừng huấn luyện. Chạy lại file để tiếp tục.")

# --- 4. ĐIỂM BẮT ĐẦU CHẠY SCRIPT ---
if __name__ == "__main__":
    main()