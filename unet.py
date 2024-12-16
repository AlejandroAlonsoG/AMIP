import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Possible to add here some regularization (Batch normalization, dropout...)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            # Possible to add here some regularization (Batch normalization, dropout...)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes, class_mapping, input_channels=3):

        super(UNet, self).__init__()

        self.class_mapping = class_mapping

        self.encoder1 = ConvBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bridge = ConvBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_1_output = self.encoder1(x)
        encoder_1_pooling = self.pool1(encoder_1_output)

        encoder_2_output = self.encoder2(encoder_1_pooling)
        encoder_2_pooling = self.pool2(encoder_2_output)

        encoder_3_output = self.encoder3(encoder_2_pooling)
        encoder_3_pooling = self.pool3(encoder_3_output)

        encoder_4_output = self.encoder4(encoder_3_pooling)
        encoder_4_pooling = self.pool4(encoder_4_output)

        bridge_output = self.bridge(encoder_4_pooling)

        decoder_4_upconv = self.upconv4(bridge_output)
        decoder_4_output = self.decoder4(torch.cat([decoder_4_upconv, encoder_4_output], dim=1))

        decoder_3_upconv = self.upconv3(decoder_4_output)
        decoder_3_output = self.decoder3(torch.cat([decoder_3_upconv, encoder_3_output], dim=1))

        decoder_2_upconv = self.upconv2(decoder_3_output)
        decoder_2_output = self.decoder2(torch.cat([decoder_2_upconv, encoder_2_output], dim=1))

        decoder_1_upconv = self.upconv1(decoder_2_output)
        decoder_1_output = self.decoder1(torch.cat([decoder_1_upconv, encoder_1_output], dim=1))

        return self.out(decoder_1_output)
    
    # Using softmax and the einsum is for ensuring no losing the gradients, otherwise the loss function cannot work
    def compress_unet_output(self, output):
        probabilities = torch.softmax(output, dim=1) # Gets the predictions on 1D

        tensor_mapping = torch.tensor(self.class_mapping, device=output.device).float()
        compressed_output = torch.einsum('bchw,c->bhw', probabilities, tensor_mapping)

        return compressed_output
