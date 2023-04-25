import torch
import torch.nn as nn
from tqdm import tqdm
from datamanager import *
import torch.optim as optim
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

from DeepSense_HAR import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"WORKING WITH {DEVICE}")

model = DeepSense().to(DEVICE)
PATH  = './dataset/HAR_UCI'
trainset, testset = DM(f'{PATH}/train'), DM(f'{PATH}/test')
LR = 0.0001
OPTIM = optim.Adam(model.parameters(), lr=LR)
BATCH_SIZE = 32
criterion  = nn.CrossEntropyLoss()
EPOCH = 100


train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True , drop_last=True)
test_loader  = DataLoader(testset , batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


for epoch in tqdm(range(EPOCH), ascii=True):
    loss_trace = []
    model.train()
    for idx, batch in enumerate(tqdm(train_loader, leave=False)):
        OPTIM.zero_grad()
        X, Y = batch
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        pred = model(X)
        loss = criterion(pred, Y)
        loss_trace.append(loss.cpu().detach().numpy())
        loss.backward()
        OPTIM.step()
        
    model.eval()
    with torch.no_grad():
        loss_trace = []
        result_pred, result_anno = [], []
        for idx, batch in enumerate(test_loader):
            X, Y = batch
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred, Y)
            loss_trace.append(loss.cpu().detach().numpy())
            pred_np  = pred.to('cpu').detach().numpy()
            pred_np  = np.argmax(pred_np, axis=1).squeeze()
            Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
            result_pred = np.hstack((result_pred, pred_np))
            result_anno = np.hstack((result_anno, Y_np))
        acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
    
    print(f'\n({epoch+1:03}/{EPOCH}) ACC:{acc*100}')