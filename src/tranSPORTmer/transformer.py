import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        
        Q = self.W_q(Q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Mask is [B, seq_len] containing 1.0 for padding/nan
            # We expand to [B, 1, 1, seq_len] to mask out invalid keys
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 1.0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, mask=None):
        attn_out = self.mha(X, X, X, mask=mask)
        X = X + self.dropout(attn_out)
        X = self.ln1(X)
        ffn_out = self.ffn(X)
        X = X + self.dropout(ffn_out)
        return self.ln2(X)

class TemporalSAB(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.sab = SetAttentionBlock(d_model, num_heads, d_ff, dropout)
    
    def forward(self, X, mask=None):
        B, T, N, D = X.shape
        X_flat = X.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        # Reformulate mask for temporal dimension
        if mask is not None:
            flat_mask = mask.permute(0, 2, 1).reshape(B * N, T)
        else:
            flat_mask = None
            
        X_flat = self.sab(X_flat, mask=flat_mask)
        return X_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)

class SocialSAB(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.sab = SetAttentionBlock(d_model, num_heads, d_ff, dropout)
    
    def forward(self, X, mask=None):
        B, T, N, D = X.shape
        X_flat = X.reshape(B * T, N, D)
        
        # Reformulate mask for social dimension
        if mask is not None:
            flat_mask = mask.reshape(B * T, N)
        else:
            flat_mask = None
            
        X_flat = self.sab(X_flat, mask=flat_mask)
        return X_flat.reshape(B, T, N, D)

class TranSPORTmer(nn.Module):
    def __init__(self, input_dim=3, d_model=128, num_heads=8, num_encoder_layers=2, d_ff=512, dropout=0.1, num_agents=3, seq_len=125, target_seq_len=35):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max(seq_len, target_seq_len))
        
        self.coarse_encoder = nn.ModuleList([
            TemporalSAB(d_model, num_heads, d_ff, dropout),
            TemporalSAB(d_model, num_heads, d_ff, dropout),
            SocialSAB(d_model, num_heads, d_ff, dropout)
        ])
        self.fine_encoder = nn.ModuleList([
            TemporalSAB(d_model, num_heads, d_ff, dropout),
            TemporalSAB(d_model, num_heads, d_ff, dropout),
            SocialSAB(d_model, num_heads, d_ff, dropout)
        ])
        
        # Project 125 sequence length -> 35 target length future prediction
        self.temporal_project = nn.Linear(seq_len, target_seq_len)
        self.output_head = nn.Linear(d_model, input_dim)
    
    def forward(self, X, mask=None):
        B, T, N, C = X.shape
        
        # CRITICAL SAFETY: To prevent NaN from exploding during PyTorch matmul, zero out -inf.
        # The true representation of missing tokens is now handled strictly via the attention mask.
        X_safe = torch.where(torch.isinf(X) | torch.isnan(X), torch.zeros_like(X), X)
        
        X_emb = self.embedding(X_safe)
        X_emb = X_emb + self.pos_encoding(X_emb.view(B * T, N, -1)).view(B, T, N, -1)
        
        J = X_emb
        for layer in self.coarse_encoder:
            J = layer(J, mask=mask)
        for layer in self.fine_encoder:
            J = layer(J, mask=mask)
        
        # Project temporally [B, T, N, D] -> [B, N, D, T] -> [B, N, D, target_T]
        J_perm = J.permute(0, 2, 3, 1) 
        J_proj = self.temporal_project(J_perm)
        J_out = J_proj.permute(0, 3, 1, 2)  # Back to [B, target_T, N, D]
        
        X_pred = self.output_head(J_out)
        return X_pred

def create_model(config=None):
    if config is None:
        config = {
            'input_dim': configs['tranSPORTmer']['model']['input_dim'],
            'd_model': configs['tranSPORTmer']['model']['d_model'],
            'num_heads': configs['tranSPORTmer']['model']['num_heads'],
            'num_encoder_layers': configs['tranSPORTmer']['model']['num_encoder_layers'],
            'd_ff': configs['tranSPORTmer']['model']['d_ff'],
            'dropout': configs['tranSPORTmer']['model']['dropout'],
            'num_agents': configs['tranSPORTmer']['model']['num_agents'],
            'seq_len': configs['tranSPORTmer']['model']['seq_len'],
            'target_seq_len': configs['tranSPORTmer']['model']['target_seq_len']
        }
    return TranSPORTmer(**config)