import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, channels):
        super(UNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(channels, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool1d(10),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool1d(8),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.enc4 = nn.Sequential(
            nn.MaxPool1d(6),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.enc5 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_1 = nn.Sequential(
            nn.Upsample(18),
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(113),
            nn.Conv1d(128, 64, 6, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(903),
            nn.Conv1d(64, 32, 8, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.decoder_4 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(9005),
            nn.Conv1d(32, 16, 10, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.decoder_5 = nn.Sequential(
            nn.Conv1d(32, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.segment1 = nn.Sequential(
            nn.Conv1d(16, 5, 1), nn.BatchNorm1d(5), nn.Tanh(),
        )

        self.segment2 = nn.Sequential(nn.Conv1d(5, 1, 1), nn.Sigmoid())

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        x = self.decoder_1(enc5)
        x = torch.cat((x, enc4), 1)

        x = self.decoder_2(x)
        x = torch.cat((x, enc3), 1)

        x = self.decoder_3(x)
        x = torch.cat((x, enc2), 1)

        x = self.decoder_4(x)
        x = torch.cat((x, enc1), 1)

        x = self.decoder_5(x)
        x = self.segment1(x)

        x = x.view(-1, 5, 100, 90)
        x = torch.mean(x, dim=2)

        x = self.segment2(x)

        return x


if __name__ == "__main__":
    model = UNet(channels=8)
    n_params = sum(p.numel() for p in model.parameters())
    print(f" Nombre de parameters : {n_params}")
