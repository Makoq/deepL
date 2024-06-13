import torch
from const import ngpu

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
