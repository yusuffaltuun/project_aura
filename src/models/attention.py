import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class SelfAttention (nn.Module):
    def __init__(self, emb_dim: int, head_dim: int, dropout_rate: float = 0.1 , max_len: int = 1024):
        super().__init__()
        self.head_dim = head_dim
        self.W_query = nn.Linear(emb_dim, head_dim, bias=False)
        self.W_key = nn.Linear(emb_dim, head_dim, bias=False)
        self.W_value = nn.Linear(emb_dim, head_dim, bias=False)
        
        
        self.dropout = nn.Dropout(dropout_rate)


        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )
    
    def forward(self,x):
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        attn_scores = queries @ keys.transpose(1, 2)

        T = x.size(1)

        attn_scores = attn_scores.masked_fill(self.mask[0, 0, :T, :T] == 0, -1e9)
        attn_weights = F.softmax(
            attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec













if __name__ == "__main__":
 
    B, T, C = 2, 4, 6
    head_dim = 6  
    
  
    x = torch.randn(B, T, C)
    print("Girdi Boyutu (X):", x.shape)
    
  
    attention = SelfAttention(emb_dim=C, head_dim=head_dim)
    
   
    out = attention(x)
    print("Çıktı Boyutu (Out):", out.shape)