import sys
import pandas as pd
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from scipy.io.arff import loadarff

def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in tqdm(dataset):
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

model_path, data_path = sys.argv[1,2]
with open(data_path) as f:
  data = loadarff(f)
df = pd.DataFrame(data[0])
df = df.sample(frac=1.0)
df = df.drop(labels='target', axis=1)
dataset, _, _ = create_dataset(df)
model = torch.load(model_path)
print(predict(model, dataset))