# %%
from torch import nn
# from torchvision import datasets, transforms

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(60*160*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4*63),
            Reshape(4, 63)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4*63),
            Reshape(4, 63)
        )

    def forward(self, x):
        logits = self.cnn_relu_stack(x)
        return logits
    
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.crnn_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            # 因為圖片大小太小，無法使用此層
            # nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            # nn.ReLU(),
        )
        self.Flatten = nn.Linear(512*1, 64)
        self.LSTM1 = nn.LSTM(64, 256, bidirectional=True)
        self.LSTM2 = nn.LSTM(2*256, 256, bidirectional=True)
        self.output = nn.Linear(2*256, 63)

    def forward(self, x):
        x = self.crnn_stack(x)
        
        batch, channel, height, weight = x.size()
        x = x.view(batch, channel*height, weight)
        x = x.permute(2, 0, 1)
        x = self.Flatten(x)
        result, _ = self.LSTM1(x)
        result, _ = self.LSTM2(result)
        logits = self.output(result)
        
        return logits
    
# %%
def test_code():
    X = next(iter(dataloader))[0]
    X = X.reshape(20, 1, 60, 160).to('cpu')

    crnn_stack = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 256, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(1,2), stride=2),
        nn.Conv2d(256, 512, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        # 因為圖片大小太小，無法使用此層
        # nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
        # nn.ReLU(),
    )

    pred = crnn_stack(X)
    print(pred.shape)
    batch, channel, height, weight = pred.size()
    pred = pred.view(batch, channel*height, weight)
    print(pred.shape)
    pred = pred.permute(2, 0, 1)
    print(pred.shape)
    f = nn.Linear(channel*height, 64)
    pred = f(pred)
    print(pred.shape)
    LSTM1 = nn.LSTM(64, 256, bidirectional=True)
    pred, _ = LSTM1(pred)
    LSTM2 = nn.LSTM(2*256, 256, bidirectional=True)
    pred, _ = LSTM2(pred)
    output = nn.Linear(2*256, 63)
    pred = output(pred)