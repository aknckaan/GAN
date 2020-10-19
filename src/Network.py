from src.Layers import *
import torch
class Generator(torch.nn.Module):
	def __init__(self): # in 1x4x4
		super(Generator,self).__init__()
		t1 = transpose_conv_2d(channel_in=1, channel_out=8, stride=2, kernel_size=4, padding=1) # 8x16x16
		t1_res = [residual_transpose_conv_2d(channel_in=8) for i in range(4)]
		t2 = transpose_conv_2d(channel_in=8, channel_out=16, stride=2, kernel_size=4, padding=1) # 16x16x16
		t2_res = [residual_transpose_conv_2d(channel_in=16) for i in range(4)]
		t3 = transpose_conv_2d(channel_in=16, channel_out=3, stride=2, kernel_size=4, padding=1, last=True)# 4x32x32
		self.layer = torch.nn.Sequential(t1, *t1_res, t2, *t2_res, t3)

		self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-6)

	def forward(self,x):
		x = self.layer(x)
		return x 

class Critic(torch.nn.Module):
	def __init__(self): # in 3x32x32
		super(Critic,self).__init__()
		t1 = conv_2d(channel_in=3, channel_out=16, stride=2, kernel_size=4, padding=1) # 16x16x16
		t2 = conv_2d(channel_in=16, channel_out=32, stride=2, kernel_size=4, padding=1) # 32x18x8
		t3 = conv_2d(channel_in=32, channel_out=64, stride=2, kernel_size=4, padding=1, last=True)# 64x4x4
		self.linear = torch.nn.Linear(64*4*4, 1)
		self.relu= torch.nn.LeakyReLU(2)
		self.layer = torch.nn.Sequential(t1, t2, t3)

		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-5)

	def forward(self,x):
		x = self.layer(x)
		x = self.relu(self.linear(x.flatten(start_dim=1)))
		return x 
