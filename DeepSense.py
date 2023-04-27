import torch
import torch.nn as nn

CONV_NUM = 64
CONV_LEN = 3
DROPOUT_RATE = 0.2
CONV_MERGE_LENS = [8, 6, 4]

class IndConvBlock(nn.Module):
    def __init__(self):
        super(IndConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_NUM,  kernel_size=(1, 2*3*CONV_LEN), stride=(1,6), padding=(1, 2*3))
        self.conv2 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_LEN), stride=1, padding=(0, CONV_LEN//2))
        self.conv3 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_LEN), stride=1, padding=(0, CONV_LEN//2))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(num_features=CONV_NUM)
        self.dropout   = nn.Dropout(p=DROPOUT_RATE)
        
    def forward(self, X):
        X = self.conv1(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.conv2(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.conv3(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1, X.shape[3]))
        return X
    
class MerConvBlock(nn.Module):
    def __init__(self):
        super(MerConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(1, 2, CONV_MERGE_LENS[0]), stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(1, 2, CONV_MERGE_LENS[1]), padding=1)
        self.conv3 = nn.Conv3d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(1, 2, CONV_MERGE_LENS[2]), padding=1)
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm3d(num_features=CONV_NUM)
        self.dropout   = nn.Dropout(p=DROPOUT_RATE)
    def forward(self, X):
        X = self.dropout(X)
        X = self.conv1(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.conv2(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.conv3(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        return X
    
class DeepSense(nn.Module):
    def __init__(self):
        super(DeepSense, self).__init__()
        self.accIndConv  = IndConvBlock()
        self.gyroIndConv = IndConvBlock()
        self.mergeConv   = MerConvBlock()
        self.GRU         = nn.GRU(input_size=384, hidden_size=120, num_layers=2, dropout=0.5)
        self.classifier  = nn.Linear(in_features=2400, out_features=6)
    def forward(self, X):
        '''
        X: input tensor shape (B, C, 20, 120) => C is 1 for the HHAR data
        '''
        acc  = self.accIndConv( X[:,:, :, :60])
        gyro = self.gyroIndConv(X[:,:, :, 60:])
        X = torch.cat((acc, gyro), dim=3)
        # print(f"1 {X.shape}")
        X = self.mergeConv(X)
        # print(f"2 {X.shape}")
        X = X.reshape(X.shape[0], -1)
        X = X.reshape(X.shape[0], 20, -1)
        # X = X.reshape(X.shape[0], X.shape[1], -1)
        # print(f"3 {X.shape}")
        X = self.GRU(X)[0]
        # print(f"4 {X.shape}")
        X = X.reshape(X.shape[0], -1)
        # print(f"5 {X.shape}")
        X = self.classifier(X)
        return X

if __name__ == '__main__':
    print("DeepSense Implementation with PyTorch")
    model = DeepSense()
    x = torch.rand((10,1,20,120))
    out = model(x)
    print(out.shape)