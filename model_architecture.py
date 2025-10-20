import torch
import torch.nn as nn
import math

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout_prob: float, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = DynamicPositionalEncoding(d_model, dropout_prob, max_len)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x) * math.sqrt(self.d_model)
        x_emb = x_emb.transpose(0, 1)
        x_emb = self.positional_encoding(x_emb)
        return x_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_prob=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_prob):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_prob):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_prob)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, mask):
        x2 = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.feed_forward(x)
        x = self.norm2(x + self.dropout(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_prob):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_prob)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout_prob)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(x2))
        x2 = self.feed_forward(x)
        x = self.norm3(x + self.dropout(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout_prob, max_len):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, dropout_prob, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_prob) for _ in range(n_layers)
        ])
    
    def forward(self, src, src_mask):
        x = self.embedding(src).transpose(0, 1)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout_prob, max_len):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, dropout_prob, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_prob) for _ in range(n_layers)
        ])
    
    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.embedding(tgt).transpose(0, 1)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad_token_id = config.pad_idx
        self.start_token_id = config.start_token_id
        self.end_token_id = config.end_token_id
        
        self.encoder = Encoder(config.src_vocab_size, config.d_model, config.n_layers,
                              config.n_heads, config.d_ff, config.dropout_prob, config.max_len)
        self.decoder = Decoder(config.tgt_vocab_size, config.d_model, config.n_layers,
                              config.n_heads, config.d_ff, config.dropout_prob, config.max_len)
        self.output_linear = nn.Linear(config.d_model, config.tgt_vocab_size)
    
    def _create_padding_mask(self, seq):
        return (seq != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def greedy_decode(self, src, max_len=None):
        if max_len is None:
            max_len = self.config.max_len
        
        batch_size = src.size(0)
        device = src.device
        src_mask = self._create_padding_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        decoder_input = torch.full((batch_size, 1), self.start_token_id, 
                                  dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = self._create_padding_mask(decoder_input) & torch.tril(
                torch.ones((decoder_input.size(1), decoder_input.size(1)), device=device)
            ).bool().unsqueeze(0).unsqueeze(1)
            
            decoder_output = self.decode(decoder_input, encoder_output, src_mask, tgt_mask)
            logits = self.output_linear(decoder_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            if (next_token == self.end_token_id).all():
                break
        
        return decoder_input

class ModelConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
