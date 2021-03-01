import torch
import torch.nn as nn
import torch.nn.functional as F


N_FILTERS1 = 20
KERNEL_SIZE = 21

N_FILTERS2 = 40

N_FILTERS3 = 20

N_FILTERS4 = 40

KERNEL_SIZE5 = 11
DROPOUT = 0.2


class CNNet(nn.Module):

    def __init__(self):
        super(CNNet, self).__init__()
        # 1D Convolutional Net for X -> input size (9000, 8)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=N_FILTERS1, kernel_size=KERNEL_SIZE, stride=1, padding=10), # use same padding,
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.MaxPool1d(10)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(N_FILTERS1, N_FILTERS2, KERNEL_SIZE, stride=1, padding=10), # kernel_size = 5,
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.MaxPool1d(10)
        )

        # 1D Convolutional Net for X_fft -> input size (90, 90)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=90, out_channels=N_FILTERS3, kernel_size=KERNEL_SIZE, stride=1, padding=10),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.conv4 = nn.Conv1d(in_channels=N_FILTERS2+N_FILTERS3, out_channels=N_FILTERS4, kernel_size=KERNEL_SIZE, stride=1, padding=10)
        self.conv5 = nn.Conv1d(in_channels=N_FILTERS4, out_channels=1, kernel_size=KERNEL_SIZE5, stride=1, padding=5)


    def forward(self, x, x_fft):
        x = self.conv1(x)
        x = self.conv2(x)

        x_fft = self.conv3(x_fft)

        x_combined = torch.cat((x, x_fft), dim=1)
        x_combined = F.relu(self.conv4(x_combined))
        x_combined = self.conv5(x_combined) # no activation function for use with BCEWithLogitsLoss
        return x_combined.squeeze()

