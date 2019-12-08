# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:40:23 2019

@author: hp
"""

import torch
import torch.nn as nn

# 输入维度10，输出为20，1层
rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, bidirectional=True)
# 数据 input(seq_len, batch, input_size)
data = torch.randn(5, 3, 10)
# h0(num_layers * num_directions, batch, hidden_size)
# c0(num_layers * num_directions, batch, hidden_size)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)

# output(seq_len, batch, hidden_size * num_directions)
# hn(num_layers * num_directions, batch, hidden_size)
# cn(num_layers * num_directions, batch, hidden_size)
output, (hn,cn) = rnn(data, (h0, c0))

print("双向LSTM",output.shape)