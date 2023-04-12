import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Functions import Funcs_for_train as F
from sklearn.metrics import mean_squared_error
import numpy as np

### 튜닝 해야되는 부분 #####################################################################
# Load the saved model
model_name = "trained_model_230412-22-11"  # example
model_path = "models/" + model_name + ".pt"

### Data Loading and Preprocessing:
folder_kapecvaleo = 'Data/22.10.05 Raw Data/'
file_kapecvaleo = '1b.xlsx' # 다른 데이터를 넣어도 되는지는 모르겠음.
sheet_kapecvaleo = 'record'

### Prepare the data_kapecvaleo => 추후 복수 파일에 있는 데이터 싹 긁어야함.
data_kapecvaleo = F.prepare_data(folder_kapecvaleo, file_kapecvaleo, sheet_kapecvaleo)
data_kapecvaleo = F.preprocess(data_kapecvaleo)

# Example haper-parameter -> 바꿔가면서 뭐가 좋은지 찾아야함.
input_size = 1
transformer_hidden_size = 16
num_transformer_layers = 2
lstm_hidden_size = 16
output_size = 1
learning_rate = 0.001
epochs = 100
###########################################################################################

# Create a new model instance and load the state dictionary
model = F.TransformerLSTM(input_size, transformer_hidden_size, num_transformer_layers, lstm_hidden_size, output_size)
model.load_state_dict(torch.load(model_path))


### Create input sequences and targets
input_sequences = data_kapecvaleo['Current'].values.reshape(-1, 1)
targets = data_kapecvaleo['Voltage'].values.reshape(-1, 1)

### Split the data_kapecvaleo into training and validation (and optionally, testing) sets
train_data, valid_data, train_targets, valid_targets = train_test_split(input_sequences, targets, test_size=0.2, random_state=42)

# Create a DataLoader for the validation or test dataset
valid_dataset = F.BatteryDataset(torch.tensor(valid_data, dtype=torch.float32), torch.tensor(valid_targets, dtype=torch.float32))
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Evaluate the model
model.eval()
predictions = []
targets = []

with torch.no_grad():
    for inputs, target in valid_loader:
        output = model(inputs)
        predictions.extend(output.numpy().flatten())
        targets.extend(target.numpy().flatten())

# Calculate evaluation metrics
mse = mean_squared_error(targets, predictions)
rmse = np.sqrt(mse)

print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)

### Add more evaluation metrics
# Accuracy: the proportion of correct predictions out of all predictions.
# Precision: the proportion of true positive predictions out of all positive predictions. Precision is used to evaluate the fraction of relevant instances among the retrieved instances.
# Recall: the proportion of true positive predictions out of all actual positive instances. Recall is used to evaluate the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
# F1 Score: the harmonic mean of precision and recall, which gives a balance between the two metrics.
# Area Under the Curve (AUC): the area under the ROC (Receiver Operating Characteristic) curve, which is a plot of the true positive rate (TPR) against the false positive rate (FPR) for various thresholds.
# Mean Absolute Error (MAE): the average absolute difference between the predicted values and the actual values.
# Mean Squared Error (MSE): the average squared difference between the predicted values and the actual values.
# Root Mean Squared Error (RMSE): the square root of the MSE.
# R-squared (R2): a metric that measures how well the predicted values fit the actual values.
# Confusion Matrix: a table that shows the number of correct and incorrect predictions made by the model.