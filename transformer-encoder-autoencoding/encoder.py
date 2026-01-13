import torch
import torch.nn as nn
from attention import MultiHeadSelfAttention
from positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_len, num_classes=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.classifier = nn.Linear(d_model, num_classes) if num_classes else None

    def forward(self, x, attn_mask=None, task='mlm'):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
        x = self.norm(x)
        if task == 'mlm':
            return self.mlm_head(x), attn
        elif task == 'cls' and self.classifier:
            # Use [CLS] token (assume first token)
            return self.classifier(x[:, 0, :]), attn
        else:
            raise ValueError('Unknown task or classifier not defined')

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attn_mask=None):
        attn_out, attn_weights = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights
