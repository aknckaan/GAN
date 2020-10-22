import torch


class gen_residual_2d(torch.nn.Module):
	def __init__(self, channel_in, channel_out, kernel_size, stride, padding, last = False):
		super(gen_residual_2d,self).__init__()
		self.to_RGB = conv_2d(channel_in, 3, 1, 1, 0, use_spectral=False)
		self.up_sample = torch.nn.Upsample(scale_factor=2)
		self.t_conv = transpose_conv_2d(channel_in, channel_out, kernel_size, stride, padding, last)
		self.batchnorm = torch.nn.BatchNorm2d(3)
		self.sigmoid1=torch.nn.Sigmoid()

	def forward(self, x, up_images, first=False, last = False):

		if not first:
			rgb_x = self.to_RGB(x) + up_images 
		else:
			rgb_x = self.to_RGB(x)

		
		if not last:
			rgb_x =self.batchnorm(rgb_x)
		up=self.up_sample(rgb_x)
		self.sigmoid1(up)		
		processed = self.t_conv(x)
		return up, processed



class residual_transpose_conv_2d(torch.nn.Module):

	def __init__(self,channel_in,use_activation=True):
		super(residual_transpose_conv_2d,self).__init__()
		self.conv_1 = transpose_conv_2d(padding=[0,0], channel_in=channel_in,channel_out=int(channel_in*2), kernel_size=[1,1], stride = [1,1])
		self.conv_2 = transpose_conv_2d(padding=[1,1], channel_in=int(channel_in*2),channel_out=channel_in, kernel_size=[3,3], stride = [1,1])

		self.batchnorm = torch.nn.BatchNorm2d(channel_in)
		# self.batchnorm = torch.nn.InstanceNorm2d(channel_in)
	
		self.use_activation = use_activation
		self.activation = torch.nn.LeakyReLU(2)
		
		
	def forward(self,x):
		x_=self.conv_1(x)
		x_=self.conv_2(x_)

		x = x+x_
		x=self.batchnorm(x)
		x=self.activation(x)
		return x
		

class transpose_conv_2d(torch.nn.Module):
	def __init__(self,channel_in, channel_out, kernel_size, stride, padding, last = False):
		super(transpose_conv_2d,self).__init__()

		conv = torch.nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, padding=padding, kernel_size=kernel_size, stride=stride)
		torch.nn.init.normal_(conv.weight,  0.0, 0.02)
		batch_norm = torch.nn.BatchNorm2d(channel_out)
		relu = torch.nn.LeakyReLU(2)
	
		if last :
			sigmoid = torch.nn.Sigmoid()
			self.layer = torch.nn.Sequential(conv, sigmoid)
		else:
			relu = torch.nn.LeakyReLU(2)
			self.layer = torch.nn.Sequential(conv, batch_norm, relu)
	
	def forward(self, x):
		x=self.layer(x)
		return x

class conv_2d(torch.nn.Module):
	def __init__(self,channel_in, channel_out, kernel_size, stride, padding, last=False, use_spectral=True):
		super(conv_2d,self).__init__()

		if use_spectral:
			conv =  torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=channel_in, out_channels=channel_out, padding=padding, kernel_size=kernel_size, stride=stride))
		else:
			conv =  torch.nn.Conv2d(in_channels=channel_in, out_channels=channel_out, padding=padding, kernel_size=kernel_size, stride=stride)
		
		torch.nn.init.normal_(conv.weight,  0.0, 0.02)
		batch_norm = torch.nn.BatchNorm2d(channel_out)
		relu = torch.nn.LeakyReLU(2)
		
		if last :
			self.layer = torch.nn.Sequential(conv, relu)
		else:
			self.layer = torch.nn.Sequential(conv, batch_norm, relu)
	
	def forward(self, x):
		x=self.layer(x)
		return x