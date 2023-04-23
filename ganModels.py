import torch.nn
import torch.nn as nn
#from torchsummary import summary

#  out_size = （in_size - K + 2P）/ S +1

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, 8, 4, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(4, 8, 8, 4, 1, bias=False),
            torch.nn.BatchNorm1d(8),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(8, 16, 8, 4, 1, bias=False),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(16, 32, 8, 4, bias=False),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, 7, 4, 1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 8, 4, 1, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 10, 8, 4, bias=False),
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(110, 50),
            torch.nn.Linear(50, 10),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x.view(x.size(0), 1, -1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.linear(x.view(x.size(0), -1))
        return x



class generator(nn.Module):
    def __init__(self, noise_length):
        super(generator, self).__init__()

        self.dconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(128, 128, 8, 4, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True)
        )
        self.dconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(128, 64, 8, 4, 1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(True)
        )
        self.dconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(64, 32, 7, 4, 1, bias=False),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(True)
        )
        self.dconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(32, 16, 8, 4, bias=False),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(True)
        )
        self.dconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(16, 8, 8, 4, 1, bias=False),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(True)
        )
        self.dconv6 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(8, 4, 8, 4, 1, bias=False),
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(True)
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(4, 1, 8, 4, 1, bias=False),

        )
    def forward(self, x):
        x = x.view(x.size(0), 128, 11)
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        x = self.dconv5(x)
        x = self.dconv6(x)
        x = self.dconv7(x)
        x = x.view(x.size(0), -1)
        return x