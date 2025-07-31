# 台語語音辨識系統（Taiwanese Hokkien Speech-to-Text）

本專案旨在提供一套完整且可擴展的台語（臺灣閩南語）語音辨識解決方案，涵蓋從語音 ➜ 拼音 ➜ 漢字的雙階段架構，以及基於 LoRA 微調的 Whisper 模型，並同時支援本地與雲端部署。整體技術棧採用 PyTorch、Hugging Face Transformers、PEFT、Accelerate 等先進工具，確保訓練效能、推論效率與易用性。

🔗 **線上體驗**：[Hugging Face Spaces - KikKoh/Hokkien](https://huggingface.co/spaces/KikKoh/Hokkien)

---

## 🎯 專案亮點

* **雙階段架構**：

  * **Stage 1**：台語語音 ➜ 羅馬拼音（台羅） (my-wav2vec2 模組)
  * **Stage 2**：羅馬拼音 ➜ 台語漢字 (hok2han 模組)
* **基於 LoRA 微調的 Whisper 模型**：
  * **Mode 1**：台語語音 ➜ 台語漢字 (lora-whisper 模組)
  * **Mode 2**：台語語音 ➜ 中文文字 (lora-whisper-zh 模組)
* **多模型支援**：CTC-Based (Wav2Vec2)、Transformer Seq2Seq、Whisper + LoRA 微調
* **高效訓練**：混合精度 (AMP)、LoRA 參數高效微調、Accelerate 分散式訓練
* **易用部署**：Hugging Face Hub / Spaces 一鍵上傳、Dockerfile
* **開放原始碼**：Apach 2.0，歡迎學術與非商業用途

---

## 📂 專案結構

```
taiwanese-speech-to-text/
├── data/
│   ├── 詞條音檔/...
│   ├── 例句音檔/...
│   └── kautian.ods
│
├── model/
│   ├── my-wav2vec2/...
│   ├── hok2han/...
│   ├── lora-whisper/...
│   └── lora-whisper-zh/...
│
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📥 安裝與環境準備

1. **Clone 專案**

   ```bash
   git clone https://github.com/KikKoh/taiwanese-speech-to-text.git
   cd taiwanese-speech-to-text
   ```
2. **建立虛擬環境**（建議使用 `venv` 或 `conda`）

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```
3. **安裝相依套件**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **(可選) 安裝 GPU / Accelerate 支援**

   ```bash
   pip install accelerate
   accelerate config  # 初始化設定
   ```

---

## 🚀 快速上手

### 1. 推論：語音 ➜ 羅馬拼音

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("my-wav2vec2")
model = Wav2Vec2ForCTC.from_pretrained("my-wav2vec2").to(device)
model.eval()

waveform, sr = sf.read("audio.wav")
waveform = torch.tensor(waveform).float()
if waveform.dim() == 1:
  waveform = waveform.unsqueeze(0)
else:
  waveform = waveform.permute(1, 0)
if waveform.shape[0] > 1:
  waveform = waveform.mean(dim=0, keepdim=True)
if sr != target_sample_rate:
  resampler = torchaudio.transforms.Resample(sr, 16000)
  waveform = resampler(waveform)

audio_input = processor(waveform, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(audio_input.input_values).logits
pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
romaji = processor.batch_decode(pred_ids)
print("羅馬拼音：", romaji)
```

### 2. 推論：拼音 ➜ 台語漢字

```python
from transformers import AutoTokenizer
from model.hok2han import Seq2SeqTransformer

input_tokenizer = AutoTokenizer.from_pretrained("KikKoh/Hok2Han", subfolder="input_tokenizer")
output_tokenizer = AutoTokenizer.from_pretrained("KikKoh/Hok2Han", subfolder="output_tokenizer")
model = Seq2SeqTransformer.from_pretrained("KikKoh/Hok2Han").to(device)
model.eval()

encoded = input_tokenizer("pinyin", max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

start_token_id = output_tokenizer.cls_token_id or output_tokenizer.convert_tokens_to_ids('<s>')
end_token_id = output_tokenizer.sep_token_id or output_tokenizer.convert_tokens_to_ids('</s>')

tgt_ids = torch.tensor([[start_token_id]], device=device)
total_confidence = 0.0
token_count = 0

for _ in range(max_len - 1):
  tgt_mask = generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
  tgt_key_padding_mask = (tgt_ids == output_tokenizer.pad_token_id).to(device)

  outputs = model(input_ids, tgt_ids, input_tokenizer.pad_token_id, =output_tokenizer.pad_token_id,
            src_key_padding_mask=(attention_mask == 0), tgt_key_padding_mask=tgt_key_padding_mask)
  next_token_logits = outputs[:, -1, :]
  probs = softmax(next_token_logits, dim=-1)
  next_token = torch.argmax(probs, dim=-1).unsqueeze(1)
  tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
  if next_token.item() == end_token_id:
    break
translation = output_tokenizer.decode(tgt_ids[0], skip_special_tokens=True).replace(" ", "")
print("台語漢字：", translation)
```

### 3. 推論：Whisper + LoRA(zh)

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="zh", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(model, "demo/lora-whisper").to(device).eval()

waveform, sr = sf.read("audio.wav")
waveform = torch.tensor(waveform).float()
if waveform.dim() == 1:
  waveform = waveform.unsqueeze(0)
else:
  waveform = waveform.permute(1, 0)
if waveform.shape[0] > 1:
  waveform = waveform.mean(dim=0, keepdim=True)
if sr != target_sample_rate:
  resampler = torchaudio.transforms.Resample(sr, 16000)
  waveform = resampler(waveform)

inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
with torch.no_grad():
  generated_ids = model.generate(input_features=inputs.input_features.to(device), task="transcribe", language="zh")
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("生成文本：", transcription)
```

---

## 🛠️ 自訂訓練

各子模組已提供完整 `README.md`，範例訓練流程包含：

* **my-wav2vec2**：CTC 微調、AMP、線性 warm-up scheduler
* **hok2han**：自架 Seq2Seq Transformer、CrossEntropyLoss、Early Stopping
* **lora-whisper / lora-whisper-zh**：LoRA 微調、Accelerate 分散式訓練、WER 驗證

請參考對應資料夾下的 `README.md`，並依據 GPU 計算資源、語料規模調整超參數。

---

## 🤝 貢獻指南

歡迎台語愛好者、語音處理研究者、AI 開發者參與：

1. Fork 本倉庫並建立分支 (`git checkout -b feature/xxx`)
2. 完成開發後提交 PR，詳述修改內容與測試結果
3. Issue 中提案或討論新功能

---

## 📜 授權條款

* **程式碼**：MIT License ([LICENSE](./LICENSE))
* **語料資料**：依據中華民國教育部《臺灣閩南語常用詞辭典》CC BY-ND 3.0 TW 條款，僅用於學術研究與非商業用途

---

## ✉️ 聯絡方式

如有疑問，請透過 GitHub Issue 或私訊聯絡：

* GitHub: [@10809104](https://github.com/10809104)
* Hugging Face Spaces: [KikKoh/Hokkien](https://huggingface.co/spaces/KikKoh/Hokkien)
* facebook: [KikKoh2024](https://www.facebook.com/kikkoh.2024))

---

祝研究順利，期待您的貢獻！
