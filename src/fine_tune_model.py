import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from jiwer import wer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data.dataset import ASRDataset, custom_collate_fn
from src.data.tokenizer import Tokenizer


# ==========================================
# 🧠 MODEL: Transformer-based CTC Encoder
# ==========================================
class CTCModel(nn.Module):
    def __init__(self, input_dim, vocab_size,
                 d_model=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.linear = nn.Linear((input_dim // 4) * 64, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, x_len):
        B, T, F = x.size()
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, T_, F_ = x.size()
        x = x.transpose(1, 2).contiguous().view(B, T_, C * F_)
        x = self.linear(x)
        x = self.encoder(x)
        logits = self.classifier(x)
        return logits, T_


# ==========================================
# 🔡 GREEDY DECODE
# ==========================================
def greedy_decode(logits, tokenizer):
    pred_ids = torch.argmax(logits, dim=-1)
    texts = []
    for seq in pred_ids:
        prev = -1
        tokens = []
        for idx in seq.cpu().numpy():
            if idx != prev and idx != 0:
                tokens.append(tokenizer.id2token[idx])
            prev = idx
        texts.append("".join(tokens))
    return texts


# ==========================================
# 🏋️ TRAIN ONE EPOCH
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, tokenizer, scaler):
    model.train()
    total_loss = 0.0
    wers = []

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        inputs = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        input_lens = batch["input_lengths"] // 4
        label_lens = batch["label_lengths"]

        optimizer.zero_grad()

        # ✅ AMP dùng đúng cách
        with torch.amp.autocast("cuda"):
            logits, _ = model(inputs, input_lens)
            logits_ctc = logits.transpose(0, 1)

        # ✅ Tính loss ở float32 (CTCLoss không hỗ trợ half)
        with torch.autocast(device_type="cuda", enabled=False):
            loss = criterion(logits_ctc.float(), labels, input_lens, label_lens)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # ✅ Tính WER cho vài batch đầu để theo dõi
        if i < 5:
            pred_texts = greedy_decode(logits, tokenizer)
            wers.extend([wer(ref.lower(), hyp.lower())
                         for ref, hyp in zip(batch["text"], pred_texts)])

    avg_loss = total_loss / len(dataloader)
    avg_wer = sum(wers) / len(wers) if wers else None
    return avg_loss, avg_wer


# ==========================================
# 🚀 MAIN TRAIN LOOP
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    train_manifest = os.path.join(project_root, "data", "manifests", "train_360.csv")
    vocab_path = os.path.join(project_root, "src", "data", "vocab.json")
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = ASRDataset(train_manifest, vocab_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    tokenizer = Tokenizer(vocab_path=vocab_path)
    vocab_size = len(tokenizer.vocab)

    model = CTCModel(input_dim=80, vocab_size=vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda")  # ✅ Sửa cảnh báo mới

    # Resume checkpoint nếu có
    start_epoch = 0
    resume_path = os.path.join(ckpt_dir, "ctc_model_360_epoch156.pt")
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resume training từ epoch {start_epoch}")

    # Huấn luyện
    end_epoch = start_epoch + 30
    for epoch in range(start_epoch, end_epoch):
        avg_loss, avg_wer = train_one_epoch(model, dataloader, criterion, optimizer, device, tokenizer, scaler)
        if avg_wer is not None:
            print(f"✅ Epoch {epoch+1} | Loss = {avg_loss:.4f} | WER ~ {avg_wer:.3f}")
        else:
            print(f"✅ Epoch {epoch+1} | Loss = {avg_loss:.4f}")

        # ✅ Save checkpoint an toàn
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
        }
        ckpt_path = os.path.join(ckpt_dir, f"ctc_model_360_epoch{epoch+1}.pt")
        torch.save(ckpt_data, ckpt_path)
        torch.save(ckpt_data, os.path.join(ckpt_dir, "ctc_model_360_last.pt"))
        print(f"💾 Saved checkpoint: {ckpt_path}")
