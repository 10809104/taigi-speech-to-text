---
library_name: pytorch, transformers
base_model: KikKoh/Hok2Han
license: apache-2.0
tags:
- seq2seq
- transformer
- taiwanese
- pinyin-to-chinese
---

# Model Card for Hok2Han

本模型為基於 PyTorch 實作的 Seq2Seq Transformer，用於將台語拼音轉換成台語漢字。

## Model Details

### Model Description

本模型利用自訂架構 Seq2Seq Transformer，學習從台語拼音序列映射到對應的台語漢字序列。訓練資料包含大量台語語料及對應拼音標註，模型架構包含6層編碼器與解碼器，採用512維嵌入與8頭注意力機制。

* **Developed by:** KikKoh
* **Model type:** Seq2Seq Transformer
* **Language(s):** Taiwanese (Hokkien)
* **License:** Apache-2.0
* **Finetuned from model:** 自訂架構，非標準預訓練模型

### Model Sources

* **Repository:** [https://huggingface.co/KikKoh/Hok2Han](https://huggingface.co/KikKoh/Hok2Han)
* **Config and weights:** Hugging Face Hub

## Uses

### Direct Use

可用於台語拼音轉漢字的自動翻譯、語音識別後處理等應用場景。

### Out-of-Scope Use

不適用於非台語拼音輸入、其他語言翻譯或語音直接識別。

## Bias, Risks, and Limitations

模型僅訓練於台語拼音資料，對其他方言、口音或非標準拼音可能表現不佳。使用時應注意語料多樣性限制及可能產生誤翻譯。

## How to Get Started with the Model

```python
from hok2han_model import Seq2SeqTransformer
model = Seq2SeqTransformer.from_pretrained("KikKoh/Hok2Han")
model.eval()

from transformers import Wav2Vec2Processor, BertTokenizer
input_processor = Wav2Vec2Processor.from_pretrained("你的輸入tokenizer路徑或repo")
output_tokenizer = BertTokenizer.from_pretrained("你的輸出tokenizer路徑或repo")

# 範例推論
output = model(src=input_ids, tgt=tgt_ids,
               src_pad_idx=input_processor.tokenizer.pad_token_id,
               tgt_pad_idx=output_tokenizer.pad_token_id)

pred_ids = output.argmax(dim=-1)
pred_text = output_tokenizer.decode(pred_ids[0], skip_special_tokens=True)
print(pred_text)
```

## Training Details

### Training Data

使用台語拼音與漢字對照語料，包含公開及自建資料。

### Training Procedure

使用標準Seq2Seq Transformer訓練方法，採用交叉熵損失，AdamW優化器。

## Evaluation

評估主要依據拼音到漢字的轉換準確率及語句流暢度。

## Environmental Impact

訓練過程使用標準GPU伺服器，耗電與碳排放量中等。

## Technical Specifications

### Model Architecture and Objective

6層編碼器與解碼器，512維嵌入，8頭多頭注意力。

---

歡迎聯絡 KikKoh
Facebook: [https://www.facebook.com/kikkoh.2024](https://www.facebook.com/kikkoh.2024)

---