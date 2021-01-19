import torch.nn as nn
import torch
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        # 三个卷积层用于提取特征
        # 1 input channel image 214*214, 8 output channel image  106*106
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 8 input channel image 106x106, 16 output channel image 22x22
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 16 input channel image 22x22, 32 output channel image 10x10
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        # 分类
        self.classifier = nn.Sequential(
           # nn.Linear(32 * 10 * 10, 2)
            nn.Linear(64 * 7 * 7, 512),
           # nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
       # print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
      # x=  self.conv5(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
       # print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x

if __name__ == '__main__':
    model = SimpleNet()
    print(model)
    inputs = torch.randn(4,3,150,150)
    output = model(inputs)
    #a, preds = torch.max(output, 1)
    #print(a)
   #print(preds)
    #print(output)
    print(output.shape)