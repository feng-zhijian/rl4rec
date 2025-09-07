import torch
import torch.nn as nn
import torch.nn.functional as F

class NextItNetResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, causal=True):
        super(NextItNetResidualBlock, self).__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) if causal else (dilation * (kernel_size - 1)) // 2
        )
        self.ln1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=2 * dilation,
            padding=(2 * dilation * (kernel_size - 1)) if causal else (2 * dilation * (kernel_size - 1)) // 2
        )
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        x_ = x.transpose(1, 2)  # (batch, channels, seq_len)
        out = self.conv1(x_)
        if self.causal:
            out = out[:, :, :x_.size(2)]  # remove extra padding for causal
        out = out.transpose(1, 2)  # (batch, seq_len, channels)
        out = self.ln1(out)
        out = F.relu(out)
        out_ = out.transpose(1, 2)
        out = self.conv2(out_)
        if self.causal:
            out = out[:, :, :x_.size(2)]
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = F.relu(out)
        return x + out
