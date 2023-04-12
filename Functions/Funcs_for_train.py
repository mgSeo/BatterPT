import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def prepare_data(folder_path, file_path, file_sheet):

    # Read all sheets at once
    data = pd.read_excel(folder_path + file_path, sheet_name=file_sheet, engine='openpyxl')

    # Function to remove units from column names
    def remove_units(column_name):
            return column_name.split("(")[0].strip()

    # Remove units from all column names
    data.columns = [remove_units(col) for col in data.columns]

    # Convert time columns to datetime format
    data['Time'] = pd.to_timedelta(data['Time'].astype(str))
    data['Total time'] = pd.to_timedelta(data['Total time'].astype(str))
    data['Real time'] = pd.to_datetime(data['Real time'], format="%Y-%m-%d %H:%M:%S")

    return data


def preprocess(data):

    # Remove unnecessary columns
    cols_to_drop = ['Data serial number', 'Cycle ID', 'Step ID', 'Step Type', 'Time', 'Total time', 'Specific Capacity', 'Real time', 'Module start-stop switch', 'dQ/dV', 'dQm/dV']
    data = data.drop(columns=cols_to_drop)

    # Handle missing values (if any)
    data = data.dropna()

    # Normalize the numerical features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

class BatteryDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TransformerLSTM(nn.Module):
    def __init__(self, input_size, transformer_hidden_size, num_transformer_layers, lstm_hidden_size, output_size):
        super(TransformerLSTM, self).__init__()

        self.transformer_hidden_size = transformer_hidden_size

        # Transformer layers
        self.positional_encoding = PositionalEncoding(transformer_hidden_size)
        transformer_layer = TransformerEncoderLayer(d_model=transformer_hidden_size, nhead=4)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=transformer_hidden_size, hidden_size=lstm_hidden_size, batch_first=True)

        # Output layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add the sequence length dimension
        x = x.to(torch.float)  # Convert x to float
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x, _ = self.lstm(x)
        x = self.fc(x.squeeze(1))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def train(train_loader, valid_loader, input_size, transformer_hidden_size, num_transformer_layers, lstm_hidden_size, output_size, learning_rate, epochs):
    model = TransformerLSTM(input_size, transformer_hidden_size, num_transformer_layers, lstm_hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch: {epoch+1}/{epochs}, Training loss: {train_loss:.5f}')

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Validation Loss: {valid_loss / len(valid_loader)}')
        
    return model