import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, channels, dropout):
        super(UNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(channels, 8, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool1d(10),
            nn.Conv1d(8, 10, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Conv1d(10, 10, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool1d(8),
            nn.Conv1d(10, 12, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.Conv1d(12, 12, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
        )


        self.decoder_3 = nn.Sequential(
            nn.Conv1d(12, 12, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.Conv1d(12, 12, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.Upsample(903),
            nn.Conv1d(12, 10, 8, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
        )

        self.decoder_4 = nn.Sequential(
            nn.Conv1d(10 * 2, 10, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Conv1d(10, 10, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Upsample(9005),
            nn.Conv1d(10, 8, 10, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.decoder_5 = nn.Sequential(
            nn.Conv1d(8 * 2, 8, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 5, padding=2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.segment1 = nn.Sequential(
            nn.Conv1d(8, 4, 1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(4),
            nn.Tanh())

        self.segment2 = nn.Sequential(
            nn.Conv1d(4, 1, 1),
            nn.Sigmoid())

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        x = self.decoder_3(enc3)
        x = torch.cat((x, enc2), 1)

        x = self.decoder_4(x)
        x = torch.cat((x, enc1), 1)

        x = self.decoder_5(x)
        x = self.segment1(x)

        x = x.view(-1, 4, 100, 90)
        x = torch.mean(x, dim=2)

        x = self.segment2(x)

        return x


if __name__ == "__main__":
    model = UNet(channels=8, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f" Nombre de parameters : {n_params}")
