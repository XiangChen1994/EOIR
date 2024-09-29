import torch.nn as nn

class LK_encoder(nn.Module):
    def __init__(self, in_cs, out_cs, kernel_size=5, stride=1, padding=2):
        super(LK_encoder, self).__init__()
        self.in_cs = in_cs
        self.out_cs = out_cs
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.regular = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, 3, 1, 1),
            nn.InstanceNorm3d(out_cs),
        )
        self.large = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_cs),
        )
        self.one = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, 1, 1, 0),
            nn.InstanceNorm3d(out_cs),
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x1 = self.regular(x)
        x2 = self.large(x)
        x3 = self.one(x)
        if self.in_cs == self.out_cs and self.stride == 1:
            x = x1 + x2 + x3 + x
        else:
            x = x1 + x2 + x3
        return self.prelu(x)

class encoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=3, stride=1, padding=1):
        super(encoder, self).__init__()
        if kernel_size == 3:
            self.layer = nn.Sequential(
                nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding),
                nn.InstanceNorm3d(out_cs),
                nn.PReLU()
            )
        elif kernel_size > 3:
            self.layer = LK_encoder(in_cs, out_cs, kernel_size, stride, padding)

    def forward(self, x):
        return self.layer(x)

class decoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose3d(in_cs, out_cs, kernel_size, stride, padding, output_padding),
            nn.InstanceNorm3d(out_cs),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)