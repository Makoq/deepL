import torch
import torch.nn as nn

# 创建一个转置卷积层
conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

# 输入特征图
input_tensor = torch.tensor([[[[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]]]], dtype=torch.float32)

# 执行前向传播
output_tensor = conv_transpose(input_tensor)

print(output_tensor)
