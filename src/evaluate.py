import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import jiwer  # Thư viện để tính WER (cài bằng: pip install jiwer)
import kenlm
import numpy as np
from pyctcdecode import build_ctcdecoder
import pandas as pd
from collections import Counter

# --- 1. THIẾT LẬP ĐƯỜNG DẪN (PATH SETUP) ---
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import từ các file .py trong 'src'
try:
    from src.data.dataset import ASRDataset, custom_collate_fn 
    from src.data.tokenizer import Tokenizer
    from src.models.asr_model import ASRModel
except ImportError as e:
    print(f"LỖI: Không thể import từ thư mục 'src'.")
    print(f"Lỗi gốc: {e}")
    sys.exit(1)

# ===================================================================
# CẤU HÌNH
# ===================================================================
# --- Đường dẫn ---
# (!!) QUAN TRỌNG: Chúng ta evaluate trên file 'dev.csv'
MANIFEST_PATH = os.path.join(project_root, "data", "manifests", "dev.csv")
VOCAB_PATH = os.path.join(project_root, "data", "tokenizer_vocab.json")
CHECKPOINT_PATH = os.path.join(project_root, "checkpoints", "ctc_model_last.pt")
LM_PATH = os.path.join(project_root, "lm", "4gram.bin") 
MANIFEST_TRAIN_PATH = os.path.join(project_root, "data", "manifests", "train.csv") # Để lấy unigrams

# --- Cấu hình Model (Phải giống hệt lúc train) ---
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
N_MELS = 80
SAMPLE_RATE = 16000

# --- Cấu hình Evaluate ---
BATCH_SIZE = 16  # Có thể tăng lên (ví dụ: 32) khi evaluate nếu GPU đủ VRAM
NUM_WORKERS = 4  # (Chỉnh về 0 nếu bị lỗi trên Windows)

# ===================================================================
# HÀM CHÍNH (MAIN FUNCTION)
# ===================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Thiết bị sử dụng: {device}")

    # --- 1. Tải Tokenizer và Unigrams ---
    print(f"Đang tải Tokenizer từ: {VOCAB_PATH}")
    tokenizer = Tokenizer(vocab_path=VOCAB_PATH)
    vocab_list = []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.id2token[i]
        if token not in (tokenizer.PAD_TOKEN, tokenizer.BLANK_TOKEN):
            vocab_list.append(token)

    print(f"Đang tải unigrams từ file manifest: {MANIFEST_TRAIN_PATH}")
    try:
        df = pd.read_csv(MANIFEST_TRAIN_PATH)
        all_text = " ".join(df["text"].tolist())
        words = all_text.split()
        unigrams = Counter(words)
        print(f"Đã tìm thấy {len(unigrams)} unigrams (từ vựng).")
    except Exception as e:
        print(f"Lỗi khi tải unigrams: {e}. Bỏ qua...")
        unigrams = None

    # --- 2. Khởi tạo Beam Search Decoder ---
    print(f"Khởi tạo Beam Search Decoder và tải KenLM từ: {LM_PATH}")
    try:
        beam_search_decoder = build_ctcdecoder(
            vocab_list,
            kenlm_model_path=LM_PATH,
            unigrams=unigrams,
            alpha=0.5,
            beta=1.5,
        )
    except Exception as e:
        print(f"Lỗi: Không thể tải file KenLM. Đảm bảo file tồn tại ở: {LM_PATH}")
        print(f"Lỗi gốc: {e}")
        sys.exit(1)

    # --- 3. Khởi tạo Model ASR ---
    model = ASRModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        n_mels=N_MELS
    ).to(device)

    # --- 4. Tải Checkpoint ---
    print(f"Tải checkpoint từ: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval() # <-- QUAN TRỌNG: Chuyển sang chế độ evaluation

    # --- 5. Tải Dữ liệu (Tập DEV) ---
    print(f"Đang tải Dataset (dev) từ: {MANIFEST_PATH}")
    eval_dataset = ASRDataset(
        manifest_path=MANIFEST_PATH, 
        vocab_path=VOCAB_PATH,
        n_mels=N_MELS,
        use_spec_augment=False # <-- QUAN TRỌNG: Tắt augmentation khi evaluate
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # <-- QUAN TRỌNG: Không xáo trộn
        collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"✅ Tải Dataset (dev) thành công với {len(eval_dataset)} mẫu.")

    # --- 6. Vòng lặp Đánh giá ---
    all_predictions = []
    all_references = []

    print("🚀 Bắt đầu đánh giá (Evaluate)...")
    
    # Tắt tính toán gradient
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if batch is None:
                continue
                
            inputs = batch["input_features"].to(device)
            ref_texts = batch["text"] # Lấy text gốc (ground truth)
            
            # Chạy model (dùng AMP cho nhanh)
            with torch.amp.autocast("cuda"):
                log_probs = model(inputs) # (T', B, Vocab)
            
            log_probs_permuted = log_probs.permute(1, 0, 2) # (B, T', Vocab)
            log_probs_numpy = log_probs_permuted.cpu().numpy()

            # Sắp xếp lại log_probs cho pyctcdecode
            log_probs_blank = log_probs_numpy[:, :, 1:2] 
            log_probs_chars = log_probs_numpy[:, :, 2:]
            log_probs_decoder = np.concatenate((log_probs_blank, log_probs_chars), axis=2)

            # Chạy decode cho từng mẫu trong batch
            for i in range(log_probs_decoder.shape[0]):
                logits_item = log_probs_decoder[i] # Lấy mẫu thứ i
                beams = beam_search_decoder.decode_beams(logits_item, beam_width=100)
                pred_text = beams[0][0] # Lấy kết quả tốt nhất
                
                all_predictions.append(pred_text)
                all_references.append(ref_texts[i])

    # --- 7. Tính toán & Báo cáo WER ---
    print("✅ Đánh giá hoàn tất. Bắt đầu tính WER...")
    
    wer = jiwer.wer(all_references, all_predictions)
    
    print("\n" + "="*30)
    print(f"🏁 KẾT QUẢ ĐÁNH GIÁ (Checkpoint: {CHECKPOINT_PATH})")
    print(f"Word Error Rate (WER): {wer * 100:.2f} %")
    print("="*30 + "\n")

    # In ra một vài ví dụ
    print("--- Ví dụ ---")
    for i in range(min(10, len(all_predictions))):
        print(f"Ref (gốc):      {all_references[i]}")
        print(f"Pred (dự đoán): {all_predictions[i]}\n")


if __name__ == "__main__":
    main()