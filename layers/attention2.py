import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention2(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, n_head=1, dropout=0.1):
        super(Attention2, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)  # Added
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)  # Added
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
        
        # k, q linear
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)  # (bs, k_len, n_head, hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)  # (bs, q_len, n_head, hidden_dim)
        
        # multi-head attention
        kx = kx.transpose(1, 2).contiguous().view(-1, k_len, self.hidden_dim)  # (n_head*bs, k_len, hidden_dim)
        qx = qx.transpose(1, 2).contiguous().view(-1, q_len, self.hidden_dim)  # (n_head*bs, q_len, hidden_dim)
        
        qw = torch.matmul(qx, self.weight) # (n_head*bs, q_len, hidden_dim)
        
        score = F.tanh(torch.bmm(qw, kx.permute(0, 2, 1))) # (n_head*bs, q_len, k_len)
        attentions = F.softmax(score, dim=-1)
        
        # Mask
        mask = torch.ones((score.size(0), 1, score.size(2))).cuda()
        for i, l in enumerate(memory_len): 
            mask[i, :, l:] = 0
        masked = attentions * mask
        _sums = masked.sum(-1)  # sums per row
        attentions = torch.div(masked, _sums.unsqueeze(-1))

        output = torch.bmm(attentions, kx)  # (n_head*bs, q_len, hidden_dim)
        output = output.view(self.n_head, mb_size, q_len, self.hidden_dim)  # (n_head, bs, q_len, hidden_dim)
        output = output.transpose(1, 0).contiguous().view(mb_size, q_len, -1)  # (bs, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, embed_dim)
        output = self.dropout(output)
        return output, attentions
