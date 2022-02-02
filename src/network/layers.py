import torch


class UpsampleLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super(UpsampleLayer, self).__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.LeakyReLU(1e-2),
        )
        self.layer0_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels * 2, in_channels * 2, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.LeakyReLU(1e-2),
        )
        self.conv_t1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels * 2, in_channels * 2, 3, 2, 1, 1, bias=False
            ),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.LeakyReLU(0.2),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels * 2, in_channels * 2, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer0_1(x)
        x = self.conv_t1(x)
        x = self.layer1(x)

        return x
