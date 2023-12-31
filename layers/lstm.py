import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False):
        
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.LSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  

    def forward(self, x, x_len):
        
        """sort"""
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx)
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=self.batch_first)
        out_pack, (ht, ct) = self.LSTM(x_emb_p, None)
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        """unpack: out"""
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
        # out = out[0]  #
        out = out[x_unsort_idx]
        """unsort: out c"""
        ct = torch.transpose(ct, 0, 1)[x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ct = torch.transpose(ct, 0, 1)

        return out, (ht, ct)
