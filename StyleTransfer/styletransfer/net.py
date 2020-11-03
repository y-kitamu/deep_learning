import torch
import torch.nn as nn

from styletransfer.function import calc_mean_std, adaptive_instance_normalization


class VGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, x):
        return self.decoder(x)


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        self.features = nn.Sequential(*model.features[:21])

    def forward(self, inputs):
        return self.features(inputs)


class Net(nn.Module):
    def __init__(self, encoder=VGGEncoder(), decoder=VGGDecoder(), alpha=1.0):
        super(Net, self).__init__()
        enc_layers = list(encoder.features.children())
        self.enc1 = nn.Sequential(*enc_layers[:2])
        self.enc2 = nn.Sequential(*enc_layers[2:7])
        self.enc3 = nn.Sequential(*enc_layers[7:12])
        self.enc4 = nn.Sequential(*enc_layers[12:21])
        self.decoder = decoder
        self.adain = adaptive_instance_normalization
        # self.adain = adain.AdaIN()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

        # for name in ['enc1', 'enc2']:
        #     for param in getattr(self, name).parameters():
        #         param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, contents, styles):
        content_feat = self.encode(contents)
        style_feats = self.encode_with_intermediate(styles)
        trans_feat = self.adain(content_feat, style_feats[-1])
        trans_feat = self.alpha * trans_feat + (1 - self.alpha) * content_feat
        decoded = self.decoder(trans_feat)
        decoded_feats = self.encode_with_intermediate(decoded)

        content_loss = self.calc_content_loss(decoded_feats[-1], trans_feat)
        style_loss = self.calc_style_loss(decoded_feats[0], style_feats[0])
        for i in range(1, len(style_feats)):
            style_loss += self.calc_style_loss(decoded_feats[i], style_feats[i])

        return content_loss, style_loss
