import torch
import torch.nn as nn

CONV_NUM = 32
CONV_LEN = 3
DROPOUT_RATE = 0.2
CONV_MERGE_LENS = [3, 3, 3]

class IndConvBlock(nn.Module):
    def __init__(self):
        super(IndConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_NUM,  kernel_size=(3, CONV_LEN), stride=1, padding=(0, CONV_LEN//2))
        self.conv2 = nn.Conv1d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_LEN), padding=(1))
        self.conv3 = nn.Conv1d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_LEN), padding=(1))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm1d(num_features=CONV_NUM)
        self.dropout   = nn.Dropout(p=DROPOUT_RATE)
        
    def forward(self, X):
        # print("0", X.shape)
        X = self.conv1(X)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[3]))
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        # print("1", X.shape)
        X = self.conv2(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        # print("2", X.shape)
        X = self.conv3(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        # print("3", X.shape)
        X = X.reshape((X.shape[0], X.shape[1], 1, X.shape[2]))
        # print("4", X.shape)
        return X
    
class MerConvBlock(nn.Module):
    def __init__(self):
        super(MerConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(3, CONV_MERGE_LENS[0]), stride=1, padding=(0, CONV_MERGE_LENS[0]//2))
        self.conv2 = nn.Conv1d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_MERGE_LENS[1]), padding=(CONV_MERGE_LENS[1]//2))
        self.conv3 = nn.Conv1d(in_channels=CONV_NUM, out_channels=CONV_NUM,  kernel_size=(CONV_MERGE_LENS[2]), padding=(CONV_MERGE_LENS[2]//2))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm1d(num_features=CONV_NUM)
        self.dropout   = nn.Dropout(p=DROPOUT_RATE)
    def forward(self, X):
        # print("00", X.shape)
        X = self.dropout(X)
        X = self.conv1(X)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[3]))
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        # print("11", X.shape)
        X = self.conv2(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        # print("22", X.shape)
        X = self.conv3(X)
        X = self.batchNorm(X)
        X = self.relu(X)
        # print("33", X.shape)
        return X
    
class DeepSense(nn.Module):
    def __init__(self):
        super(DeepSense, self).__init__()
        self.accIndConv  = IndConvBlock()
        self.gyroIndConv = IndConvBlock()
        self.magIndConv = IndConvBlock()
        self.mergeConv   = MerConvBlock()
        self.GRU         = nn.GRU(input_size=256, hidden_size=120, num_layers=2, dropout=0.5)
        self.classifier  = nn.Linear(in_features=3840, out_features=7)
    def forward(self, X):
        '''
        X: input tensor shape (B, 1, 9, 256)
        '''
        acc  = self.accIndConv( X[:,:,  :3, :])
        gyro = self.gyroIndConv(X[:,:, 3:6, :])
        mag  = self.magIndConv( X[:,:, 6:9, :])
        X = torch.cat((acc, gyro, mag), dim=2)
        X = self.mergeConv(X)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        X = self.GRU(X)[0]
        X = X.reshape(X.shape[0], -1)
        X = self.classifier(X)
        return X

if __name__ == '__main__':
    print("DeepSense Implementation with PyTorch")
    model = DeepSense()
    x = torch.rand((10, 1, 9, 256))
    out = model(x)
    print(out.shape)