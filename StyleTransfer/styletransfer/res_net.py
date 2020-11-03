import torch.nn as nn
import torchvision.models as models

from styletransfer import net
from styletransfer.function import adaptive_instance_normalization


class ResNet18AE(nn.Module):
    def __init__(self):
        super(ResNet18AE, self).__init__()
        self.model = models.resnet18(pretrained=True)
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
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
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
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(64, 3, (7, 7)),
        )

    def forward_basic_block(self, x, basic_block, with_intermid=False):
        identity = x
        out = basic_block.conv1(x)
        if with_intermid:
            intermid = out
        out = basic_block.relu(out)
        out = basic_block.conv2(out)
        if basic_block.downsample is not None:
            identity = basic_block.downsample(x)
        out += identity
        out = basic_block.relu(out)
        if with_intermid:
            return out, intermid
        return out

    def forward_encode(self, x):
        x = self.model.conv1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.forward_basic_block(x, self.model.layer1[0])
        x = self.forward_basic_block(x, self.model.layer1[1])

        x = self.forward_basic_block(x, self.model.layer2[0])
        x = self.forward_basic_block(x, self.model.layer2[1])

        x = self.forward_basic_block(x, self.model.layer3[0])
        x = self.forward_basic_block(x, self.model.layer3[1])

        x = self.model.layer4[0].conv1(x)
        return x


class Net(net.Net):
    def __init__(self, autoencoder=ResNet18AE(), alpha=1.0):
        super(Net, self).__init__()
        self.ae = autoencoder
        self.decoder = autoencoder.decoder
        self.adain = adaptive_instance_normalization
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def encode_with_intermediate(self, x):
        x = self.ae.model.conv1(x)
        x = self.ae.model.relu(x)
        x = self.ae.model.maxpool(x)

        x, s1 = self.ae.forward_basic_block(x, self.ae.model.layer1[0], with_intermid=True)
        x = self.ae.forward_basic_block(x, self.ae.model.layer1[1])

        x, s2 = self.ae.forward_basic_block(x, self.ae.model.layer2[0], with_intermid=True)
        x = self.ae.forward_basic_block(x, self.ae.model.layer2[1])

        x, s3 = self.ae.forward_basic_block(x, self.ae.model.layer3[0], with_intermid=True)
        x = self.ae.forward_basic_block(x, self.ae.model.layer3[1])

        x = self.ae.model.layer4[0].conv1(x)
        return [s1, s2, s3, x]

    def encode(self, x):
        return self.ae.forward_encode(x)
