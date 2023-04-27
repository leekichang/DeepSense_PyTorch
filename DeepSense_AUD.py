import torch
import torch.nn as nn

CONV_NUM = 32
CONV_LEN = 3
DROPOUT_RATE = 0.2
CONV_MERGE_LENS = [3, 3, 3]

class IndConvBlock(nn.Module):
    def __init__(self):
        super(IndConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_NUM,  kernel_size=(3, CONV_LEN), padding=(1, CONV_LEN//2))
        self.conv2 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(3, CONV_LEN), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(3, CONV_LEN), padding=(1, 1))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(num_features=CONV_NUM)
        self.dropout   = nn.Dropout(p=DROPOUT_RATE)
        self.maxpool   = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, X):
        X = self.conv1(X)
        # print("0", X.shape)
        X = self.maxpool(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        # print("1", X.shape)
        # X = self.conv2(X)
        # X = self.batchNorm(X)
        # X = self.relu(X)
        # X = self.maxpool(X)
        # X = self.dropout(X)
        # print("2", X.shape)
        # X = self.conv3(X)
        # X = self.batchNorm(X)
        # X = self.relu(X)
        # print(X.shape)
        # print("3", X.shape)
        # X = X.reshape((X.shape[0], X.shape[1], 1, X.shape[2], X.shape[3]))
        # print("4", X.shape)
        return X
    
class MerConvBlock(nn.Module):
    def __init__(self):
        super(MerConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(1, CONV_MERGE_LENS[0]), stride=1, padding=(CONV_MERGE_LENS[0]//2, CONV_MERGE_LENS[0]//2))
        self.conv2 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_MERGE_LENS[1]), padding=(CONV_MERGE_LENS[1]//2))
        self.conv3 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_MERGE_LENS[2]), padding=(CONV_MERGE_LENS[2]//2))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(num_features=CONV_NUM)
        self.dropout   = nn.Dropout(p=DROPOUT_RATE)
    def forward(self, X):
        # print("00", X.shape)
        X = self.dropout(X)
        X = self.conv1(X)
        # print(X.shape)
        # X = X.reshape((X.shape[0], X.shape[1], X.shape[3], X.shape[4]))
        X = self.batchNorm(X)
        X = self.relu(X)
        # X = self.dropout(X)
        # print("11", X.shape)
        # X = self.conv2(X)
        # X = self.batchNorm(X)
        # X = self.relu(X)
        # X = self.dropout(X)
        # print("22", X.shape)
        # X = self.conv3(X)
        # X = self.batchNorm(X)
        # X = self.relu(X)
        # print("33", X.shape)
        return X
    
class DeepSense(nn.Module):
    def __init__(self):
        super(DeepSense, self).__init__()
        self.accIndConv  = IndConvBlock()
        self.mergeConv   = MerConvBlock()
        self.GRU         = nn.LSTM(input_size=5120, hidden_size=512, num_layers=2) #, dropout=0.5)
        self.classifier  = nn.Linear(in_features=5120, out_features=8)
    def forward(self, X):
        '''
        X: input tensor shape (B, 1, 60, 100)
        '''
        # print(X.shape)
        X  = self.accIndConv(X)
        # print(X.shape)
        X = self.mergeConv(X)
        # print(X.shape)
        X = X.permute(0, 3, 2, 1)
        # print(X.shape)
        X = X.reshape(X.shape[0], 10, -1)
        X = X.permute(1, 0, 2)
        X = self.GRU(X)[0]
        X = X.permute(1, 0, 2)
        X = X.reshape(X.shape[0], -1)
        X = self.classifier(X)
        return X

if __name__ == '__main__':
    print("DeepSense Implementation with PyTorch")
    model = DeepSense()
    x = torch.rand((10, 1, 60, 100))
    out = model(x)
    print(out.shape)