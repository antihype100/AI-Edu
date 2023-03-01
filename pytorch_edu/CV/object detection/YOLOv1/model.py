import torch
import torch.nn as nn
import torchvision.models as models


class YoloV1(nn.Module):
    def __init__(self, num_classes, split_size, num_boxes):
        super(YoloV1, self).__init__()

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        
        self.conv_nn = models.efficientnet_b4(weights='DEFAULT')
        self.conv_nn.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1792, out_features=(self.S * self.S * (self.C + self.B * 5)), bias=True)
        )
    
    def forward(self, x):
        out = self.conv_nn(x)
        return out

x = torch.rand(8, 3, 224, 224)

model = YoloV1(num_classes=20, split_size=7, num_boxes=2)
print(model)
print(model(x).shape)