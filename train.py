import torch
import torchvision
import torchvision.transforms as transforms
from src.Network import Generator, Critic
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
device = "cuda:2"
def main():
	sr = SummaryWriter("./logs/test_6_classes/")
	transform = transforms.Compose(
	[transforms.ToTensor()])
	trainset = torchvision.datasets.CIFAR10(root='../data/CIFAR', train=True,
										download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
										  shuffle=True, num_workers=8)

	testset = torchvision.datasets.CIFAR10(root='../data/CIFAR', train=False,
										   download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=256,
											 shuffle=False, num_workers=8)

	classes = ('plane', 'car', 'bird', 'cat',
				'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	gen = Generator().to(device)
	crit = Critic().to(device)
	for epoc in tqdm(range(1000)):
		crit_loss_arr=[]
		gen_loss_arr=[]

		for elem in (trainloader):
			data = elem[0].to(device)
			label = elem[1]

			noise_in=torch.rand((data.shape[0],16)).view([data.shape[0],1,4,4]).to(device)
			# class_in=torch.zeros((data.shape[0],len(classes),4,4)).to(device)
			# class_in[label]=torch.zeros((data.shape[0],1,4,4)).to(device)
			# noise_in=torch.cat(noise_in,class_in)
			# train discriminatior
			if epoc%2==0:
				real_crit = crit(data) # D(X)
				fake = gen(noise_in).detach() # G(z)
				fake_crit = crit(fake) # D(G(z))

				mixed_images = data.detach()*0.6 + fake.detach()*0.4
				mixed_images.requires_grad=True
				mixed_scores = crit(mixed_images)

				# gradient penalty
				gradient = torch.autograd.grad(
					inputs=mixed_images,
					outputs=mixed_scores,
					grad_outputs=torch.ones_like(mixed_scores), 
					create_graph=True,
					retain_graph=True,
				)[0]
				gradient = gradient.view(len(gradient), -1)
				gradient_norm = gradient.norm(2, dim=1)
				penalty=torch.pow((gradient_norm-1),2).mean()
				#

				crit_loss = -1*(real_crit.mean() - fake_crit.mean()) + 10*penalty
				crit.optimizer.zero_grad()
				crit_loss.backward()
				crit.optimizer.step()

			# train generator
			fake = gen(noise_in) # G(z)
			fake_crit = crit(fake) # D(G(z))
			gen_loss = -1*fake_crit.mean()
			gen.optimizer.zero_grad()
			gen_loss.backward()
			gen.optimizer.step()
			gen_loss_arr.append(gen_loss.cpu().item())
			crit_loss_arr.append(crit_loss.cpu().item())

		if epoc%2==0:
			sr.add_scalar("Crit Loss",(np.stack(crit_loss_arr)).mean(),epoc)

		sr.add_scalar("Gen Loss",(np.stack(gen_loss_arr)).mean(),epoc)

		if epoc%10==0:
			sr.add_image("fake example",fake.cpu()[0],epoc)

if __name__ == "__main__":
	main()