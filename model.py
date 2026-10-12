import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # Thông thường d_model là 512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        position = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)

        # Apply the sin to even positions
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer("pe", "pe")

    def forward(self, x):
        pass

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiply
        self.bias = nn.Paramerter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # dim -1 là gì?
        std = x.std(dim = -1, keepdim=True)
        return self.alpha* (x-mean)/(std+self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # Tại sao dùng relu, tại sao không có hàm max, tại sao dùng dropout

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "không chia hết"

        self.d_k = d_model // h
        self.w_q = nn.Linea(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1))/ math.sqrt(d_k) # (batch, h, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_weights = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_weights = dropout(attention_weights)

        return (attention_weights @ value), attention_weights # (batch, h, seq_len, d_k), (batch, h, seq_len, seq_len)


    def forward(self, q, k, v, mask):
        # mask 
        query = self.w_q(q)     # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)       # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v)     # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # (batch, h, seq_len, d_k), (batch, h, seq_len, seq_len)

        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # 1 vài implementation apply norm ở cuối. sublayer -> dropout -> plus -> norm self.norm(x + self.dropout(sublayer(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_foward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_foward_block = feed_foward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward():






