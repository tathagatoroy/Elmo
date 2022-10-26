from model import RoyNet
import torch


device = torch.device("cpu")
data = torch.rand(500,100,100)
m = RoyNet(device)

res = m(data)
