# src/models/asr_model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Module Positional Encoding (PE) cổ điển (sin/cos).
    (Thêm từ file trước)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input x: (SeqLen, Batch, D_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ASRModel(nn.Module):
    """
    CNN (giảm T x4) → Linear projection về d_model → TransformerEncoder → Linear ra vocab.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 2048,          
        n_mels: int = 80, # 80 chiều
        dropout: float = 0.1,
    ):
        super().__init__()

        # 2 conv stride=2 để giảm số frame theo thời gian T -> T/4
        # Input: (B, 1, 80, T)
        self.conv = nn.Sequential(
            # (B, 1, 80, T) -> (B, 64, 40, T/2)
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            # (B, 64, 40, T/2) -> (B, 128, 20, T/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Tính toán conv_out_dim
        # Sau conv, feature dim theo tần số còn 20 (n_mels//4), channel=128
        conv_out_dim = 128 * (n_mels // 4) # 128 * 20 = 2560

        # Thu gọn về d_model để Transformer không quá to
        self.proj = nn.Linear(conv_out_dim, d_model)

        # THÊM 2: Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, vocab_size)
        
        # THÊM 3: LogSoftmax cho CTC
        self.log_softmax = nn.LogSoftmax(dim=2) # Áp dụng trên chiều Vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, n_mels, T) - (B, 1, 80, T)
        return: (T', B, vocab_size) - Shape cho CTCLoss
        """
        # SỬA 1: Không cần unsqueeze, input đã đúng shape (B, 1, 80, T)
        x = self.conv(x)  # (B, 128, 20, T')
        
        B, C, F, T = x.shape
        # Chuyển (B, C, F, T') -> (B, T', C, F)
        x = x.permute(0, 3, 1, 2) 
        # Flatten C và F -> (B, T', C*F)
        x = x.contiguous().view(B, T, C * F)  
        
        x = self.proj(x)  # (B, T', d_model)
        
        # THÊM 2: Positional Encoding
        # Cần (T', B, D) cho PE
        x = x.permute(1, 0, 2) 
        x = self.pos_encoder(x)
        # Trả về (B, T', D)
        x = x.permute(1, 0, 2) 
        
        x = self.encoder(x)  # (B, T', d_model)
        logits = self.fc(x)  # (B, T', vocab)
        
        # THÊM 3: LogSoftmax
        log_probs = self.log_softmax(logits) # (B, T', vocab)
        
        # SỬA 3: Trả về shape cho CTC Loss
        return log_probs.permute(1, 0, 2) # (T', B, vocab)

    def get_output_lengths(self, input_lengths):
        """
        THÊM 4: Hàm tính độ dài cho CTC
        Input (B,)
        """
        # Tính toán dựa trên 2 lớp Conv2d (stride=2, padding=1, kernel=3)
        def get_len(l_in):
            l_out = (l_in + 2 * 1 - 3) // 2 + 1 # Lớp 1
            l_out = (l_out + 2 * 1 - 3) // 2 + 1 # Lớp 2
            return l_out

        # Cách đơn giản hơn (vì padding/kernel hợp lý)
        return torch.ceil(input_lengths.float() / 4).int()