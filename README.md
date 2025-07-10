# 台語語音辨識模型（Taiwanese Speech-to-Text）

本專案旨在建立一個台語（臺灣閩南語）語音轉換成中文字的深度學習模型。資料來源為教育部《臺灣台語常用詞辭典》，提供原始詞條文字與對應發音音檔，使用者可透過此模型進行台語語音輸入並取得漢字輸出。

## 📦 專案目標

- 建立台語語音 ➜ 羅馬拼音 ➜ 中文文字的兩階段模型
- 利用官方資料集訓練語音辨識模型（ASR）
- 提供 Colab / Hugging Face 使用範例
- 支援自訂資料增補與語音推論

## 📁 資料來源與授權

本專案所使用之語音與詞條文字資料來自：

- 教育部《臺灣台語常用詞辭典》
- 官方網站：https://sutian.moe.edu.tw/
- 授權條款：創用 CC 姓名標示－禁止改作 3.0 台灣（CC BY-ND 3.0 TW）  
  條款說明：https://creativecommons.org/licenses/by-nd/3.0/tw/

> ⚠ 本專案所提供之「文字與音檔」資料**未經改作**，僅用於學術研究、非商業用途。
> 原始資料版權屬於 **中華民國教育部** 所有。

## 🧰 使用工具

- Python 3.10+
- PyTorch / torchaudio
- 🤗 Transformers / Datasets
- Google Colab（訓練與推論）

## 📂 專案結構

```

taiwanese-speech-to-text/

```

## 🧪 訓練範例（開發中）

敬請期待，將支援：
- 教育部台語音檔轉 wav
- 使用 Wav2Vec2 / Whisper 等模型進行語音辨識
- 自訂字典、詞彙表與前處理

## 🤝 貢獻與聯絡

歡迎台語學習者、AI 開發者、語言研究者共同參與改進本專案。如對資料或模型使用有疑問，歡迎開 issue 討論。

## 📝 License

本倉庫程式碼以 [MIT License](LICENSE) 授權。  
資料部分依據教育部提供之 [CC BY-ND 3.0 TW](https://creativecommons.org/licenses/by-nd/3.0/tw/) 條款釋出。

```
