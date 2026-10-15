import os
import sys
import torch
import torchaudio
import torch.nn as nn

# --- THÊM DỰ ÁN VÀO SYS.PATH ---
# (Để có thể import từ thư mục 'src')
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.models.asr_model import ASRModel
from src.data.tokenizer import Tokenizer

# ===================================================================
# CẤU HÌNH (PHẢI GIỐNG HỆT LÚC TRAIN)
# ===================================================================
# Đường dẫn
CHECKPOINT_PATH = os.path.join(project_root, "checkpoints", "ctc_model_last.pt")
VOCAB_PATH = os.path.join(project_root, "data", "tokenizer_vocab.json")
# File âm thanh bạn muốn nhận diện (ví dụ: file 'test.wav' trong thư mục data)
AUDIO_TO_TEST = os.path.join(project_root, "data", "hello.mp3") 

# Cấu hình Model (phải khớp với model đã train)
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
    """
    Load một file audio, chuyển thành Mel Spectrogram 80 chiều,
    và chuẩn hóa (normalize) giống hệt như ASRDataset.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Lỗi load file: {audio_path}. Lỗi: {e}")
        return None

    # 1. Resample
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # 2. Chuyển sang Mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    # 3. Tạo Mel Spectrogram (giống hệt ASRDataset)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        win_length=int(0.025 * sample_rate),
        hop_length=int(0.010 * sample_rate),
        n_mels=n_mels,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    mel_db = db_transform(mel_transform(waveform)).squeeze(0) # (n_mels, T)

    # 4. Chuẩn hóa (Normalize)
    mean = mel_db.mean()
    std = mel_db.std()
    mel_norm = (mel_db - mean) / (std + 1e-6)
    
    # 5. Thêm batch và channel (để giống input của model)
    # (n_mels, T) -> (1, 1, n_mels, T)
    return mel_norm.unsqueeze(0).unsqueeze(0)


def decode_greedy(log_probs, tokenizer):
    """
    Giải mã CTC Greedy đơn giản (chưa dùng beam search).
    Input: log_probs (T, 1, Vocab)
    """
    # Lấy token có xác suất cao nhất ở mỗi bước thời gian
    pred_tokens = torch.argmax(log_probs.squeeze(1), dim=1)
    
    decoded_text = tokenizer.decode(pred_tokens.cpu().numpy())
    return decoded_text

# ===================================================================
# HÀM CHẠY CHÍNH (MAIN)
# ===================================================================
def run_inference(audio_path):
    
    # Quyết định chạy trên CPU hay GPU (nếu máy bạn có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Chạy nhận diện trên: {device} ---")

    # 1. Tải Tokenizer
    tokenizer = Tokenizer(vocab_path=VOCAB_PATH)
    vocab_size = tokenizer.vocab_size

    # 2. Khởi tạo Model
    model = ASRModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        n_mels=N_MELS
    ).to(device)

    # 3. Tải Checkpoint (model đã train)
    print(f"Tải checkpoint từ: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    
    # QUAN TRỌNG: Chuyển model sang chế độ .eval()
    model.eval()

    # 4. Xử lý audio
    print(f"Xử lý file audio: {audio_path}")
    spec = preprocess_audio(audio_path)
    if spec is None:
        return

    spec = spec.to(device)

    # 5. Chạy nhận diện
    print("Đang nhận diện...")
    with torch.no_grad(): # Không cần tính gradient khi inference
        log_probs = model(spec) # Output: (T', B=1, Vocab)

    # 6. Giải mã (Decode)
    transcript = decode_greedy(log_probs, tokenizer)
    
    print("\n--- KẾT QUẢ NHẬN DIỆN ---")
    print(transcript)


if __name__ == "__main__":
    run_inference(AUDIO_TO_TEST)