### Read_me ###


### General imports:
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd

### Data Loading and Preprocessing:
from Functions import DataLoadPrep
folder_HPPC = 'Data/22.10.05 Raw Data/'
file_HPPC = '1a.xlsx'
sheet_HPPC = 'record' 


data_HPPC = DataLoadPrep.load_data(folder_HPPC, file_HPPC, sheet_HPPC)
#example = DataLoadPrep.preprocess_data()



### Feature Extraction:
#extract_features()


###	Train/Test Split:
#train_test_split()



###	Model Architecture:
#TransformerModel:
    #__init__(): Initialize the model with hyperparameters.
    #forward(): Define the forward pass of the model.
    #SelfAttention: Define the self-attention mechanism.
    #PositionwiseFeedForward: Define the position-wise feed-forward layer.
    #PositionalEncoding: Implement positional encoding for the input data.


### Training and Evaluation:
#train(): Train the transformer model on the training set.
#evaluate(): Evaluate the model's performance on the testing set.
#predict(): Make predictions on new HPPC test data.


###	Utility Functions:
#compute_metrics(): Calculate evaluation metrics such as accuracy, precision, recall, and F1-score.
#save_model(): Save the trained model to a file.
#load_model(): Load a pre-trained model from a file.