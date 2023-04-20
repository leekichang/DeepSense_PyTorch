import os
import numpy as np 

DATAPATH = './sepHARData_a'
WIDE = 20
SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2

def read_csv(file):
    contents = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            contents.append(line.strip().split(','))
    return contents
    

def get_data(is_train=False):
    folder = f'{DATAPATH}/train' if is_train else f'{DATAPATH}/eval'
    files  = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    X, Y = [], []
    for file in files:
        data = read_csv(file)
        data = np.reshape(data, (-1))
        feature, label = data[:WIDE*FEATURE_DIM].reshape(WIDE, FEATURE_DIM), data[WIDE*FEATURE_DIM:].argmax()
        X.append(feature)
        Y.append(label)
    X, Y = np.array(X, dtype=np.float64), np.array(Y, dtype=np.int64)
    X = X.reshape(-1, 1, WIDE, FEATURE_DIM)
    return X, Y      

def save_data(dataset, is_train=False):
    savepath = './dataset/train' if is_train else './dataset/test'
    os.makedirs(savepath, exist_ok=True)
    X, Y = dataset
    np.save(f'{savepath}/X.npy', X)
    np.save(f'{savepath}/Y.npy', Y)
    print(f"X.npy ({X.shape}) and Y.npy ({Y.shape}) is saved!")
    

if __name__ == '__main__':
    is_train = [True, False]
    for flag in is_train:
        dataset = get_data(flag)
        save_data(dataset, flag)