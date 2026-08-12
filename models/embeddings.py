# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3x3 convolution with padding
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Builds the full resnet network
class resnet50_embeddings(nn.Module):
    def __init__(self, block, layers, num_classes=50, embedding_size=128):
        self.inplanes = 64
        super(resnet50_embeddings, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)
        self.fc_softmax = nn.Linear(1000, num_classes)
        self.fc_embedding = nn.Linear(1000, embedding_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        # NOTE (inherited behaviour, kept deliberately): the training-time
        # siblings in models/TripletResnetSoftmax.py apply a ReLU between `fc`
        # and `fc_embedding`, whereas this inference path does not. Every number
        # and every released checkpoint in the paper was produced with exactly
        # this forward pass, so it is left unchanged for reproducibility. If you
        # train new models, consider making the two paths consistent by adding
        # `x = self.relu(x)` here - and re-evaluate from scratch if you do.

        x_embedding = self.fc_embedding(x)
        x_embedding = F.normalize(x_embedding, p=2, dim=1)  # L2 normalise
        x_class = self.fc_softmax(x)  # Get class predictions

        return x_embedding, x_class

# Constructs a ResNet-50 model for inferring embeddings
def resnet50(pretrained=False, num_classes=50, ckpt_path=None, embedding_size=128, **kwargs):
    """Build the inference-time embedding network.

    `pretrained=True` loads the weights from the training checkpoint at
    `ckpt_path` (as saved by utilities/utils.py: a dict with a 'model_state'
    key). strict=False so that a checkpoint whose classification head has a
    different class count still loads - the head is unused at inference.
    """
    # Define the model
    model = resnet50_embeddings(Bottleneck, [3, 4, 6, 3], embedding_size=embedding_size, num_classes=num_classes, **kwargs)

    # Should we load saved model weights
    if pretrained:
        if not ckpt_path:
            raise ValueError("resnet50(pretrained=True) requires a ckpt_path")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights_init = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        model.load_state_dict(weights_init, strict=False)

    return model