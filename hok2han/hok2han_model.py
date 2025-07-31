import torch
import torch.nn as nn
import math
import json
from huggingface_hub import hf_hub_download

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_tokens, num_decoder_tokens, emb_size=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()

        self.encoder_embedding = nn.Embedding(num_encoder_tokens, emb_size)
        self.decoder_embedding = nn.Embedding(num_decoder_tokens, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(emb_size, num_decoder_tokens)

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = src.transpose(0, 1)  # (S, N)
        tgt = tgt.transpose(0, 1)  # (T, N)

        src_emb = self.positional_encoding(self.encoder_embedding(src))
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))

        src_key_padding_mask = (src.transpose(0, 1) == src_pad_idx)  # (N, S)
        tgt_key_padding_mask = (tgt.transpose(0, 1) == tgt_pad_idx)  # (N, T)
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)  # (T, T)

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = self.fc_out(output)  # (T, N, vocab_size)
        return output.transpose(0, 1)  # (N, T, vocab_size)

    @classmethod
    def from_pretrained(cls, repo_id, revision="main", filename_model="best_model.pth", filename_config="config.json"):
        """
        repo_id: HF Hub repo 名稱，如 "KikKoh/Hok2Han"
        revision: 分支或版本，預設是 main
        filename_model: 權重檔名
        filename_config: config 檔名
        """

        # 1. 下載 config.json
        config_path = hf_hub_download(repo_id=repo_id, filename=filename_config, revision=revision)
        with open(config_path, "r") as f:
            config = json.load(f)

        # 2. 初始化模型
        model = cls(
            num_encoder_tokens=config["num_encoder_tokens"],
            num_decoder_tokens=config["num_decoder_tokens"],
            emb_size=config.get("emb_size", 512),
            nhead=config.get("nhead", 8),
            num_encoder_layers=config.get("num_encoder_layers", 6),
            num_decoder_layers=config.get("num_decoder_layers", 6),
            dim_feedforward=config.get("dim_feedforward", 2048),
            dropout=config.get("dropout", 0.1)
        )

        # 3. 下載模型權重並載入
        model_path = hf_hub_download(repo_id=repo_id, filename=filename_model, revision=revision)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        model.eval()
        return model