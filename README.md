<div align="center">

# ASR_model: Nghiên Cứu và Xây Dựng Mô Hình Nhận Dạng Giọng Nói (ASR) Hỗ Trợ Đánh Giá Phát Âm Tiếng Anh

**End-to-End Speech Recognition System Combining CNN Subsampling, Transformer Encoder & Connectionist Temporal Classification (CTC)**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-Audio%20Processing-orange?style=flat-square)](https://pytorch.org/audio/stable/index.html)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Custom%20Tokenizer-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Đề tài Nghiên cứu Khoa học Sinh viên – Trường Đại học Thủy Lợi (Phân hiệu TP. Hồ Chí Minh)**  
**Bộ môn:** Công nghệ Thông tin | **Lớp:** S27-66TTNT & S27-66CNTT  

[Báo cáo Nghiên cứu (DOCX)](./docs/Bao_Cao_NCKH_ASR_Model.docx) • [Bộ Custom Tokenizer Ký Tự](./wav2vec2_custom_tokenizer) • [Hướng dẫn Thực nghiệm](#-hướng-dẫn-cài-đặt-và-chạy-thực-nghiệm)

</div>

---

## 📌 1. TỔNG QUAN ĐỀ TÀI

Việc học phát âm tiếng Anh chuẩn đóng vai trò quyết định trong việc nâng cao kỹ năng giao tiếp. Tuy nhiên, người học thường gặp khó khăn do thiếu phản hồi chính xác và chi tiết tức thời.

Đề tài **"Nghiên cứu mô hình ASR để xây dựng ứng dụng học phát âm tiếng Anh"** tập trung thiết kế, huấn luyện và đánh giá mô hình học sâu nhận dạng giọng nói tự động (Automatic Speech Recognition - ASR) theo hướng tiếp cận **End-to-End** kết hợp kiến trúc **CNN Subsampling + Transformer Encoder** cùng hàm mất mát **CTC (Connectionist Temporal Classification)** nhằm:
1. Xây dựng mô hình âm học nhận dạng âm vị và từ vựng có độ chính xác cao đối với ngữ điệu phát âm tiếng Anh của người Việt.
2. Tích hợp bộ giải mã kép (**Greedy Search & Beam Search**) nhằm cân bằng tối ưu giữa độ chính xác nhận dạng và độ trễ xử lý.
3. Cung cấp nền tảng tính toán sai lệch ngữ âm phục vụ mô-đun chấm điểm và đưa ra phản hồi sửa lỗi phát âm cho người học.

---

## 🏗️ 2. KIẾN TRÚC MÔ HÌNH VÀ LUỒNG NHẬN DẠNG (ASR ARCHITECTURE)

Mô hình được thiết kế theo kiến trúc lai (Hybrid Deep Learning Architecture) gồm 4 khối thành phần chính:

```text
       [ Tín hiệu âm thanh đầu vào: 80-channel Log-Mel Spectrogram ]
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ 1. Khối CNN Subsampling (2D Convolutions + Striding)        │ 
     │    Trích xuất đặc trưng cục bộ, giảm chiều thời gian 4 lần  │
     └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ 2. Lớp Linear Projection & Positional Encoding (Sin/Cos)    │ 
     │    Chiếu về không gian ẩn d_model = 512, nhúng thứ tự vị trí│
     └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ 3. Khối Transformer Encoder (8 Layers, 8 Heads, d_ff=2048)  │ 
     │    Cơ chế Multi-Head Self-Attention học ngữ cảnh toàn cục   │
     └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ 4. Lớp Phân Loại Linear Head + CTC Loss & CTC Decoding      │ 
     │    - Greedy Search (Argmax) cho tốc độ suy luận tức thời    │
     │    - Prefix Beam Search cho độ chính xác cao                │
     └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────┐
     │      Văn bản nhận dạng cuối cùng (WER / CER Evaluation)     │
     └─────────────────────────────────────────────────────────────┘
```

---

## 📂 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
ASR_model/
├── docs/                                      # Tài liệu & Báo cáo nghiên cứu khoa học học thuật
│   └── Bao_Cao_NCKH_ASR_Model.docx            # Bản báo cáo nghiên cứu chi tiết nộp Hội đồng NCKH
├── src/                                       # Mã nguồn lõi của mô hình ASR
│   ├── models/                                # Kiến trúc mô hình mạng nơ-ron
│   │   ├── asr_model.py                       # CNN + Linear Projection + Positional Encoding + Transformer Encoder
│   │   └── utils.py                           # Các hàm xử lý ma trận và tính toán độ dài chuỗi
│   ├── build_corpus.py                        # Xây dựng và tiền xử lý tập ngữ liệu huấn luyện
│   ├── train.py                               # Kịch bản huấn luyện mô hình âm học với hàm mất mát CTC
│   ├── fine_tune_model.py                     # Kịch bản tinh chỉnh (Fine-tuning) mô hình
│   ├── evaluate.py                            # Kịch bản tính toán các chỉ số lỗi WER và CER
│   ├── inference.py                           # Nhận dạng giọng nói giải mã Greedy Search (Argmax)
│   ├── inference_beam.py                      # Nhận dạng giọng nói giải mã Prefix Beam Search
│   ├── export_loss.py                         # Trích xuất nhật ký mất mát (Loss curve) phục vụ báo cáo
│   ├── delete_flac.py                         # Tiện ích dọn dẹp các tệp âm thanh trung gian
│   └── training_log_extracted.csv             # Nhật ký theo dõi quá trình hội tụ của mô hình
├── wav2vec2_custom_tokenizer/                 # Từ điển ký tự và cấu hình Tokenizer tùy biến
│   ├── vocab.json                             # Bảng mã ánh xạ ký tự ngữ âm sang Token ID
│   ├── tokenizer_config.json                  # Cấu hình Tokenizer tương thích chuẩn
│   └── special_tokens_map.json                # Định nghĩa các token đặc biệt (pad, unk, blank)
├── notebooks/                                 # Thử nghiệm trích xuất đặc trưng và trực quan hóa
├── requirements.txt                           # Các thư viện Python cần thiết
└── README.md                                  # Hướng dẫn kỹ thuật và mô tả chi tiết đề tài
```

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY THỰC NGHIỆM

### 4.1. Cài đặt môi trường
```bash
# Clone kho lưu trữ
git clone https://github.com/pquang0407/ASR_model.git
cd ASR_model

# Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Linux/macOS
# hoặc: .\venv\Scripts\activate trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 4.2. Chạy thử nghiệm nhận dạng giọng nói (Inference)
- **Phương pháp 1 – Giải mã tham lam (Greedy Search - Tốc độ cao):**
  ```bash
  python src/inference.py --audio_path data/sample.wav --model_path weights/asr_checkpoint.pt
  ```
- **Phương pháp 2 – Giải mã Beam Search (Độ chính xác cao):**
  ```bash
  python src/inference_beam.py --audio_path data/sample.wav --model_path weights/asr_checkpoint.pt --beam_width 5
  ```

### 4.3. Đánh giá chất lượng mô hình (WER / CER)
```bash
python src/evaluate.py --test_data data/test_manifest.csv --model_path weights/asr_checkpoint.pt
```

---

## 👥 5. NHÓM TÁC GIẢ & VAI TRÒ THỰC HIỆN

Đề tài Nghiên cứu Khoa học Sinh viên được bảo vệ thành công trước Hội đồng Khoa học Khoa Công nghệ Thông tin – Trường Đại học Thủy Lợi (Phân hiệu TP. Hồ Chí Minh):

| STT | Thành viên thực hiện | Lớp / Khóa | Vai trò trong đề tài |
| :-: | :--- | :---: | :--- |
| **01** | **Nguyễn Văn Phú Quang** | **S27-66TTNT** | **Trưởng nhóm Đề tài (Project Lead & Technical Supervisor)**<br>• Định hướng nghiên cứu, lựa chọn kiến trúc End-to-End (CNN + Transformer Encoder) và CTC Loss.<br>• Hướng dẫn thiết kế pipeline tiền xử lý âm thanh 80-channel Mel, bộ giải mã Beam Search và kịch bản huấn luyện.<br>• Rà soát, chuẩn hóa và hoàn thiện toàn bộ báo cáo NCKH học thuật của đề tài. |
| **02** | **Trần Anh Hải** | S27-66TTNT | Thành viên nghiên cứu (Thực thi mã nguồn, tiền xử lý dữ liệu và theo dõi thực nghiệm). |
| **03** | **Vòng Anh Khôi** | S27-66CNTT | Thành viên nghiên cứu (Thu thập mẫu giọng nói và hỗ trợ đánh giá kết quả). |

**Giáo viên Hướng dẫn:**
- **ThS. Nguyễn Thị Kim Phụng** – Giảng viên Bộ môn Công nghệ Thông tin, Trường Đại học Thủy Lợi (Phân hiệu TP. Hồ Chí Minh).

---

<div align="center">
  <sub>Kho Lưu Trữ Đề Tài Nghiên Cứu Khoa Học Sinh Viên – Trường Đại học Thủy Lợi (2026)</sub>
</div>
