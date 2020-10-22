from src.Layers import *
import torch
class Generator(torch.nn.Module):
	def __init__(self): # in 1x4x4
		super(Generator,self).__init__()
		t1 = transpose_conv_2d(channel_in=2, channel_out=8, stride=2, kernel_size=4, padding=1) # 8x16x16
		t1_res = [residual_transpose_conv_2d(channel_in=8) for i in range(4)]
		t2 = transpose_conv_2d(channel_in=8, channel_out=16, stride=2, kernel_size=4, padding=1) # 16x16x16
		t2_res = [residual_transpose_conv_2d(channel_in=16) for i in range(4)]
		t3 = transpose_conv_2d(channel_in=16, channel_out=8, stride=2, kernel_size=4, padding=1)# 8x32x32
		t4 = conv_2d(channel_in=8, channel_out=4, stride=2, kernel_size=4, padding=1)# 4x64x64
		t5 = transpose_conv_2d(channel_in=4, channel_out=3, stride=2, kernel_size=4, padding=1)# 3x32x32
		t6 = transpose_conv_2d(channel_in=3, channel_out=3, stride=1, kernel_size=1, padding=0, last=True)# 3x32x32


		self.layer = torch.nn.Sequential(t1, *t1_res, t2, *t2_res, t3, t4, t5, t6)

		self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)

	def forward(self,x):
		x = self.layer(x)
		return x 

class ResidualGenerator(torch.nn.Module):
	def __init__(self):
		super(ResidualGenerator,self).__init__()
		self.t1 = gen_residual_2d(channel_in=2, channel_out=4, stride=2, kernel_size=4, padding=1)# 4x8x8
		self.t2 = gen_residual_2d(channel_in=4, channel_out=8, stride=2, kernel_size=4, padding=1)# 8x16x16
		self.t3 = gen_residual_2d(channel_in=8, channel_out=16, stride=2, kernel_size=4, padding=1)# 16x32x32
		self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)

	def forward(self,x):
		up_img, x = self.t1(x, None, first=True)
		up_img, x = self.t2(x, up_img)
		up_img, x = self.t3(x, up_img, last=True)

		return up_img



class Critic(torch.nn.Module):
	def __init__(self): # in 3x32x32
		super(Critic,self).__init__()
		t1 = conv_2d(channel_in=3, channel_out=16, stride=2, kernel_size=4, padding=1) # 16x16x16
		t2 = conv_2d(channel_in=16, channel_out=32, stride=2, kernel_size=4, padding=1) # 32x8x8
		t3 = conv_2d(channel_in=32, channel_out=64, stride=2, kernel_size=4, padding=1) # 64x4x4
		t4 = conv_2d(channel_in=64, channel_out=128, stride=2, kernel_size=4, padding=1, last=True)# 128x2x2
		self.linear = torch.nn.Linear(512, 1)
		self.relu= torch.nn.LeakyReLU(2)
		self.layer = torch.nn.Sequential(t1, t2, t3, t4)

		self.optimizer = torch.optim.Adam(self.parameters(),lr=5e-5)

	def forward(self,x):
		x = self.layer(x)
		x = self.relu(self.linear(x.flatten(start_dim=1)))
		return x 
