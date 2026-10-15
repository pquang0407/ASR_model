# src/inference_beam.py

import os
import sys
import torch
import torchaudio
import torch.nn as nn
import kenlm
import numpy as np
from pyctcdecode import build_ctcdecoder
import pandas as pd  # <-- THÊM
from collections import Counter  # <-- THÊM

# --- THÊM DỰ ÁN VÀO SYS.PATH ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.models.asr_model import ASRModel
from src.data.tokenizer import Tokenizer

# ===================================================================
# CẤU HÌNH
# ===================================================================
CHECKPOINT_PATH = os.path.join(project_root, "checkpoints", "ctc_model_last.pt")
VOCAB_PATH = os.path.join(project_root, "data", "tokenizer_vocab.json")
LM_PATH = os.path.join(project_root, "lm", "4gram.bin") 
AUDIO_TO_TEST = os.path.join(project_root, "data", "hello.mp3") 
# Thêm đường dẫn manifest để load unigrams
MANIFEST_TRAIN_PATH = os.path.join(project_root, "data", "manifests", "train.csv")

# Cấu hình Model (phải khớp)
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
N_MELS = 80
SAMPLE_RATE = 16000

# ===================================================================
# HÀM TRỢ GIÚP (HELPER FUNCTIONS)
# ===================================================================

def preprocess_audio(audio_path, n_mels=N_MELS, sample_rate=SAMPLE_RATE):
    # (Hàm này giữ nguyên, không thay đổi)
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Lỗi load file: {audio_path}. Lỗi: {e}")
        return None
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400,
        win_length=int(0.025 * sample_rate),
        hop_length=int(0.010 * sample_rate), n_mels=n_mels,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    mel_db = db_transform(mel_transform(waveform)).squeeze(0)
    mean, std = mel_db.mean(), mel_db.std()
    mel_norm = (mel_db - mean) / (std + 1e-6)
    return mel_norm.unsqueeze(0).unsqueeze(0)

# ===================================================================
# HÀM CHẠY CHÍNH (MAIN)
# ===================================================================
def run_inference_with_lm(audio_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Chạy nhận diện trên: {device} ---")

    # 1. Tải Tokenizer
    print(f"Tải Tokenizer từ: {VOCAB_PATH}")
    tokenizer = Tokenizer(vocab_path=VOCAB_PATH)
    vocab_list = []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.id2token[i]
        if token not in (tokenizer.PAD_TOKEN, tokenizer.BLANK_TOKEN):
            vocab_list.append(token)
            
    # --- THÊM MỚI: TẢI UNIGRAMS (để sửa cảnh báo) ---
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
    # --- KẾT THÚC PHẦN MỚI ---
    
    # 2. Khởi tạo Beam Search Decoder
    print(f"Khởi tạo Beam Search Decoder và tải KenLM từ: {LM_PATH}")
    try:
        beam_search_decoder = build_ctcdecoder(
            vocab_list,
            kenlm_model_path=LM_PATH,
            unigrams=unigrams,  # <-- THÊM UNIGRAMS VÀO ĐÂY
            alpha=0.5,
            beta=1.5,
        )
    except Exception as e:
        print(f"Lỗi: Không thể tải file KenLM. Đảm bảo file tồn tại ở: {LM_PATH}")
        print(f"Lỗi gốc: {e}")
        sys.exit(1)

    # 3. Khởi tạo Model ASR
    model = ASRModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        n_mels=N_MELS
    ).to(device)

    # 4. Tải Checkpoint
    print(f"Tải checkpoint từ: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 5. Xử lý audio
    print(f"Xử lý file audio: {audio_path}")
    spec = preprocess_audio(audio_path)
    if spec is None: return
    spec = spec.to(device)

    # 6. Chạy nhận diện
    print("Đang nhận diện...")
    with torch.no_grad():
        log_probs = model(spec)
        log_probs_permuted = log_probs.permute(1, 0, 2)
        log_probs_numpy = log_probs_permuted.cpu().numpy()

    # 7. Sắp xếp lại log_probs
    log_probs_blank = log_probs_numpy[:, :, 1:2] 
    log_probs_chars = log_probs_numpy[:, :, 2:]
    log_probs_decoder = np.concatenate((log_probs_blank, log_probs_chars), axis=2)
    
    # 8. Giải mã (Decode)
    print("Đang chạy Beam Search Decode (với LM)...")
    
    # --- SỬA LỖI 2: SQUEEZE BATCH DIMENSION ---
    # pyctcdecode muốn (T, Vocab), nên chúng ta lấy [0] (batch đầu tiên)
    beams = beam_search_decoder.decode_beams(log_probs_decoder[0], beam_width=100)
    
    transcript = beams[0][0]

    print("\n--- KẾT QUẢ NHẬN DIỆN (với KenLM) ---")
    print(transcript)


if __name__ == "__main__":
    run_inference_with_lm(AUDIO_TO_TEST)