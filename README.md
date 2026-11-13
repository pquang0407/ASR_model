<div align="center">

# ASR_model: Nghiên Cứu và Xây Dựng Mô Hình Nhận Dạng Giọng Nói (ASR) Hỗ Trợ Đánh Giá Phát Âm Tiếng Anh

**Vietnamese & English Acoustic Model with Wav2Vec 2.0 / Conformer, Connectionist Temporal Classification (CTC) Loss & Beam Search Decoding**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-Speech%20Processing-orange?style=flat-square)](https://pytorch.org/audio/stable/index.html)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Đề tài Nghiên cứu Khoa học Sinh viên – Trường Đại học Thủy Lợi (Phân hiệu TP. Hồ Chí Minh)**  
**Bộ môn:** Công nghệ Thông tin | **Lớp:** S27-66TTNT & S27-66CNTT  

[Báo cáo Nghiên cứu (DOCX)](./docs/Bao_Cao_NCKH_ASR_Model.docx) • [Bộ Tokenizer Ký Tự](./wav2vec2_custom_tokenizer) • [Hướng dẫn Cài đặt & Chạy](#-hướng-dẫn-cài-đặt-và-chạy-thực-nghiệm)

</div>

---

## 📌 1. TỔNG QUAN ĐỀ TÀI

Việc học phát âm tiếng Anh chuẩn đóng vai trò quyết định trong việc nâng cao kỹ năng giao tiếp. Tuy nhiên, người học thường gặp khó khăn do thiếu phản hồi chính xác và chi tiết tức thời.

Đề tài **"Nghiên cứu mô hình ASR để xây dựng ứng dụng học phát âm tiếng Anh"** tập trung nghiên cứu, tinh chỉnh và triển khai các mô hình học sâu nhận dạng giọng nói tự động (Automatic Speech Recognition - ASR) dựa trên kiến trúc **Wav2Vec 2.0 / Conformer** kết hợp hàm mất mát **CTC (Connectionist Temporal Classification)** nhằm:
1. Xây dựng mô hình âm học nhận dạng âm vị (phoneme) và từ vựng có độ chính xác cao đối với giọng phát âm tiếng Anh của người Việt.
2. Thiết kế bộ giải mã kép (**Greedy Search & Beam Search**) giúp tối ưu hóa sự cân bằng giữa độ chính xác nhận dạng và độ trễ xử lý.
3. Cung cấp nền tảng tính toán sai lệch ngữ âm phục vụ mô-đun chấm điểm và đưa ra gợi ý sửa lỗi phát âm cho người học.

---

## 🏗️ 2. KIẾN TRÚC MÔ HÌNH VÀ LUỒNG NHẬN DẠNG (ASR PIPELINE)

```text
       [ Tín hiệu âm thanh đầu vào (.wav / .flac) ]
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │           Feature Encoder (CNNs)             │ Trích xuất đặc trưng phổ âm thanh thô
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │      Context Network (Transformer / Conformer) │ Mô hình hóa sự phụ thuộc ngữ cảnh chuỗi âm học
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │              Linear Projection               │ Ánh xạ vector đặc trưng lên tập từ điển ký tự/token
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │            CTC Loss / CTC Decoding           │ 
     │       (Greedy Search / Prefix Beam Search)   │ Khử khoảng trắng, gộp token trùng và trích xuất text
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │   Đánh giá Kết quả (WER / CER Evaluation)    │ Xuất văn bản nhận dạng & Chấm điểm phát âm
     └──────────────────────────────────────────────┘
```

---

## 📊 3. KẾT QUẢ THỬ NGHIỆM VÀ ĐÁNH GIÁ (BENCHMARK)

- **Chỉ số đánh giá độ chính xác:**
  - **WER (Word Error Rate):** Tỷ lệ lỗi cấp độ từ đạt mức cạnh tranh trên tập dữ liệu thử nghiệm người Việt nói tiếng Anh.
  - **CER (Character Error Rate):** Tối ưu hóa phản ánh chính xác từng ký tự ngữ âm giúp phát hiện lỗi sai phát âm phụ âm cuối.
- **Tối ưu suy luận:** Hỗ trợ suy luận nhanh với độ trễ thấp phù hợp tích hợp vào ứng dụng học tập di động hoặc nền tảng web.

---

## 📂 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
ASR_model/
├── docs/                                      # Tài liệu & Báo cáo nghiên cứu khoa học học thuật
│   └── Bao_Cao_NCKH_ASR_Model.docx            # Bản báo cáo nghiên cứu chi tiết nộp Hội đồng NCKH
├── src/                                       # Mã nguồn lõi của mô hình ASR
│   ├── models/                                # Kiến trúc mô hình mạng nơ-ron
│   │   ├── asr_model.py                       # Lớp định nghĩa kiến trúc mô hình CTC / Wav2Vec
│   │   └── utils.py                           # Các hàm tiện ích xử lý tensor âm thanh
│   ├── build_corpus.py                        # Xây dựng và tiền xử lý tập ngữ liệu huấn luyện
│   ├── train.py                               # Kịch bản huấn luyện mô hình từ đầu (Pre-training)
│   ├── fine_tune_model.py                     # Kịch bản tinh chỉnh (Fine-tuning) trên tập dữ liệu phát âm
│   ├── evaluate.py                            # Kịch bản tính toán các chỉ số WER và CER
│   ├── inference.py                           # Nhận dạng giọng nói giải mã Greedy Search
│   ├── inference_beam.py                      # Nhận dạng giọng nói giải mã Beam Search
│   ├── export_loss.py                         # Trích xuất nhật ký mất mát (Loss curve) phục vụ báo cáo
│   ├── delete_flac.py                         # Tiện ích dọn dẹp các định dạng tệp âm thanh trung gian
│   └── training_log_extracted.csv             # Bảng ghi nhật ký huấn luyện thực tế
├── wav2vec2_custom_tokenizer/                 # Từ điển ký tự và cấu hình Tokenizer tùy biến
│   ├── vocab.json                             # Bảng mã ánh xạ ký tự ngữ âm sang Token ID
│   ├── tokenizer_config.json                  # Cấu hình Tokenizer của HuggingFace
│   └── special_tokens_map.json                # Định nghĩa các token đặc biệt (pad, unk, blank)
├── notebooks/                                 # Thử nghiệm trích xuất đặc trưng và đồ thị huấn luyện
├── requirements.txt                           # Các thư viện Python cần thiết
└── README.md                                  # Hướng dẫn kỹ thuật và mô tả chi tiết đề tài
```

---

## 🚀 5. HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY THỰC NGHIỆM

### 5.1. Cài đặt môi trường
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

### 5.2. Chạy thử nghiệm nhận dạng giọng nói (Inference)
- **Phương pháp 1 – Giải mã tham lam (Greedy Search - Tốc độ cao):**
  ```bash
  python src/inference.py --audio_path data/sample.wav --model_path weights/asr_checkpoint.pt
  ```
- **Phương pháp 2 – Giải mã Beam Search (Độ chính xác cao):**
  ```bash
  python src/inference_beam.py --audio_path data/sample.wav --model_path weights/asr_checkpoint.pt --beam_width 5
  ```

### 5.3. Đánh giá chất lượng mô hình (Evaluation)
```bash
python src/evaluate.py --test_data data/test_manifest.csv --model_path weights/asr_checkpoint.pt
```

---

## 👥 6. NHÓM TÁC GIẢ & VAI TRÒ THỰC HIỆN

Đề tài Nghiên cứu Khoa học Sinh viên được bảo vệ thành công trước Hội đồng Khoa học Khoa Công nghệ Thông tin – Trường Đại học Thủy Lợi (Phân hiệu TP. Hồ Chí Minh):

| STT | Thành viên thực hiện | Lớp / Khóa | Vai trò trong đề tài |
| :-: | :--- | :---: | :--- |
| **01** | **Nguyễn Văn Phú Quang** | **S27-66TTNT** | **Trưởng nhóm Đề tài (Project Lead & Technical Supervisor)**<br>• Xác định mục tiêu bài toán, lựa chọn kiến trúc mạng nơ-ron ASR và hàm mất mát CTC.<br>• Định hướng thiết kế tiền xử lý dữ liệu âm thanh, thuật toán giải mã Beam Search và kịch bản huấn luyện.<br>• Rà soát và biên soạn toàn diện báo cáo học thuật hoàn chỉnh của đề tài. |
| **02** | **Trần Anh Hải** | S27-66TTNT | Thành viên nghiên cứu (Thực thi mã nguồn, tiền xử lý dữ liệu và theo dõi thực nghiệm). |
| **03** | **Vòng Anh Khôi** | S27-66CNTT | Thành viên nghiên cứu (Thu thập mẫu giọng nói và hỗ trợ đánh giá kết quả). |

**Giảng viên Hướng dẫn:**
- **ThS. Đặng Tuấn Phong** – Giảng viên Bộ môn Công nghệ Thông tin.

---

<div align="center">
  <sub>Kho Lưu Trữ Đề Tài Nghiên Cứu Khoa Học Sinh Viên – Trường Đại học Thủy Lợi (2026)</sub>
</div>
