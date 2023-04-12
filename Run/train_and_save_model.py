### General imports:
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Functions import Funcs_for_train as F
import datetime
####################

### Data Loading and Preprocessing:
folder_kapecvaleo = 'Data/22.10.05 Raw Data/'
file_kapecvaleo = '1a.xlsx'
sheet_kapecvaleo = 'record'

### Prepare the data_kapecvaleo => 추후 복수 파일에 있는 데이터 싹 긁어야함.
data_kapecvaleo = F.prepare_data(folder_kapecvaleo, file_kapecvaleo, sheet_kapecvaleo)
data_kapecvaleo = F.preprocess(data_kapecvaleo)

### Create input sequences and targets
input_sequences = data_kapecvaleo['Current'].values.reshape(-1, 1)
targets = data_kapecvaleo['Voltage'].values.reshape(-1, 1)

### Split the data_kapecvaleo into training and validation (and optionally, testing) sets
train_data, valid_data, train_targets, valid_targets = train_test_split(input_sequences, targets, test_size=0.2, random_state=42)

### Create PyTorch tensors and data_kapecvaleo loaders
train_dataset = F.BatteryDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32))
valid_dataset = F.BatteryDataset(torch.tensor(valid_data, dtype=torch.float32), torch.tensor(valid_targets, dtype=torch.float32))


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

### Train the model
input_size = 1
transformer_hidden_size = 16
num_transformer_layers = 2
lstm_hidden_size = 16
output_size = 1
learning_rate = 0.001
epochs = 100

trained_model = F.train(train_loader, valid_loader, input_size, transformer_hidden_size, num_transformer_layers, lstm_hidden_size, output_size, learning_rate, epochs)

# Save the trained model
current_time = datetime.datetime.now().strftime("%y%m%d-%H-%M")
model_path = f"models/trained_model_{current_time}.pt"
torch.save(trained_model.state_dict(), model_path)
print(f"Model saved to {model_path}")