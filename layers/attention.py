import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, n_head=1, dropout=0.1):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.w_kx = nn.Parameter(torch.Tensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.Tensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

    def forward(self, k, q, memory_len):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, bs*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, bs*q_len, embed_dim)
        
        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)  # (n_head*bs, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)  # (n_head*bs, q_len, hidden_dim)
        
        qw = torch.matmul(qx, self.weight) # (n_head*bs, q_len, hidden_dim)
        
        score = F.tanh(torch.bmm(qw, kx.permute(0, 2, 1))) # (n_head*bs, q_len, k_len)
        attentions = F.softmax(score, dim=-1)
        
        # attentions = torch.squeeze(score, dim=1)
        mask = torch.ones((score.size(0), 1, score.size(2))).cuda()
        # mask *= -1e4
        for i, l in enumerate(memory_len): 
            mask[i, :, l:] = 0
        # attentions = F.softmax(score + mask, dim=-1)
        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1)  # sums per row
        attentions = torch.div(masked, _sums.unsqueeze(-1))

        output = torch.bmm(attentions, kx)  # (n_head*bs, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (bs, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, embed_dim)
        output = self.dropout(output)
        return output, attentions