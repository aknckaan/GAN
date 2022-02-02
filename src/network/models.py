import torch
from src.network.layers import UpsampleLayer


class Generator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(0.2),
        )
        self.conv_t1 = UpsampleLayer(4)
        self.conv_t2 = UpsampleLayer(8)
        self.conv_t3 = UpsampleLayer(16)
        # self.conv_t4 = UpsampleLayer(32)
        # self.conv_t5 = UpsampleLayer(64)
        self.layer_last = torch.nn.Sequential(
            torch.nn.Conv2d(32, 3, 3, 1, 1), torch.nn.Sigmoid()
        )

        # torch.nn.ConvTranspose2d(in_channels, 4, 1, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.conv_t3(x)
        # x = self.conv_t4(x)
        # x = self.conv_t5(x)
        x = self.layer_last(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # net = torchvision.models.resnet18(pretrained=True)
        # net.conv1 = torch.nn.Conv2d(4,64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.net = torch.nn.Sequential( *list(net.children())[:8], torch.nn.Flatten(1), torch.nn.Linear(512 , 1)).to("cuda")

        net = [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 4, 3, 1, 1, bias=False),
                torch.nn.BatchNorm2d(4),
                torch.nn.LeakyReLU(0.2),
            ),  # 32
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.LeakyReLU(0.2),
            ),  # 16
            torch.nn.Sequential(
                torch.nn.Conv2d(8, 16, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.LeakyReLU(0.2),
            ),  # 8
            torch.nn.Sequential(
                torch.nn.Conv2d(16, 32, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.LeakyReLU(0.2),
            ),  # 4
            # torch.nn.Sequential(torch.nn.utils.spectral_norm(torch.nn.Conv2d(32, 64, 5, 4, 2,  bias=False)), torch.nn.BatchNorm2d(64), torch.nn.LeakyReLU(0.2)), # 1
            torch.nn.Sequential(torch.nn.Conv2d(32, 1, 1, 1, 0, bias=False)),
        ]  # 1

        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)

        return x
