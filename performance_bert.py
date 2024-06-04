import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# Ensure to include the necessary dataset and model definitions
from bert import MovieReviewDataset, DistilBertModelForRegression, collate_fn, load_dataset

# Define necessary paths and parameters
model_name = 'bert1'
checkpoint_dir = os.path.join(model_name, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
test_dir = 'aclImdb/test'
batch_size = 16
plot_path = os.path.join(model_name, 'plots', 'predictions_vs_true_values.png')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Load and preprocess the dataset
test_data, test_labels = load_dataset(test_dir)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
test_dataset = MovieReviewDataset(test_data, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Initialize the model (make sure the hyperparameters match the saved model)
model = DistilBertModelForRegression(1).to(device)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate the model on the test dataset
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for input_ids, attention_masks, labels in tqdm(test_loader, desc="Evaluating"):
        input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device).float()
        outputs = model(input_ids, attention_masks).squeeze()
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
mse = mean_squared_error(all_labels, all_preds)
mae = mean_absolute_error(all_labels, all_preds)
r2 = r2_score(all_labels, all_preds)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')

# Plotting predictions vs. true values
plt.figure(figsize=(10, 5))
plt.scatter(all_labels, all_preds, alpha=0.3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs. True Values')
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f'Plot saved to {plot_path}')

predictions = np.array(all_preds)
test_ratings = np.array(all_labels)

# Classify reviews as 'positive' (1) or 'negative' (0)
binary_predictions = np.where(predictions < 0.5, 0, 1)
binary_test_ratings = np.where(np.array(test_ratings) < 0.5, 0, 1)

conf_matrix = confusion_matrix(binary_test_ratings, binary_predictions)
conf_matrix_df = pd.DataFrame(conf_matrix, 
                      index = [ 'actual negative reviews',  'actual positive reviews'], 
                      columns = [ 'predicted negative reviews',  'predicted positive reviews'] )
print(conf_matrix_df)

tp = conf_matrix[1, 1]  # True Positives
fp = conf_matrix[0, 1]  # False Positives
fn = conf_matrix[1, 0]  # False Negatives
tn = conf_matrix[0, 0]  # True Negatives

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)

print('Precision: ', precision)
print('Recall: ', recall)
print('F1 score:', f1_score)
