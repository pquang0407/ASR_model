model_ai_asr/
│
├── data/
│   ├── raw/                # Dữ liệu gốc (LibriSpeech, custom audio)
│   ├── processed/          # Dữ liệu sau khi resample 16kHz
│   └── manifests/          # CSV metadata (train.csv, dev.csv, test.csv)
│
├── notebooks/              # Jupyter notebooks cho R&D
│   ├── preprocess.ipynb
        data_explore.ipynb
│   ├── train_ctc.ipynb
│   └── inference.ipynb
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── tokenizer.py
│   │
│   ├── models/
│   │   ├── asr_model.py
│   │   └── utils.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── checkpoints/
│   └── best_model.pt
│
├── requirements.txt
└── README.md
    environment.yml
