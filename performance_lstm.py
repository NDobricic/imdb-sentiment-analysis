import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# Ensure to include the necessary dataset and model definitions
from lstm import MovieReviewDataset, LSTMModel, collate_fn, load_dataset, tokenizer, vocab, test_data, test_labels

# Define necessary paths and parameters
model_name = 'lstm2'
checkpoint_dir = os.path.join(model_name, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
test_dir = 'aclImdb/test'
batch_size = 64

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Setting up the test data loader...')
test_dataset = MovieReviewDataset(test_data, test_labels, tokenizer, vocab)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Initialize the model (make sure the hyperparameters match the saved model)
if model_name == 'lstm2':
    embedding_dim = 75
    hidden_dim = 150
    output_dim = 1
    num_layers = 2
if model_name == 'lstm3':
    embedding_dim = 100
    hidden_dim = 200
    output_dim = 1
    num_layers = 1

model = LSTMModel(len(vocab), embedding_dim, hidden_dim, output_dim, num_layers).to(device)

# Load the checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
else:
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

# Evaluate the model
model.eval()
all_targets = []
all_predictions = []

with torch.no_grad():
    for inputs, lengths, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device).float()
        lengths = lengths.cpu()
        outputs = model(inputs, lengths).squeeze()
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(outputs.cpu().numpy())

# Calculate metrics
mse = mean_squared_error(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plotting predictions vs targets
plt.figure(figsize=(10, 5))
plt.scatter(all_targets, all_predictions, alpha=0.5)
plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], color='red')
plt.xlabel('True Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Predicted vs True Ratings')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(model_name, 'plots', 'predictions_vs_true.png'))
plt.close()


predictions = np.array(all_predictions)
test_ratings = np.array(all_targets)

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