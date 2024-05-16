#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm


# In[2]:


class MovieReviewDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, label


# In[3]:


class DistilBertModelForRegression(nn.Module):
    def __init__(self, output_dim, dropout_rate=0.2):
        super(DistilBertModelForRegression, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.distilbert.config.hidden_size, output_dim)

        # Freeze the DistilBERT layers
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        dropped_output = self.dropout(pooled_output)
        output = self.fc(dropped_output)
        return output


# In[4]:


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Set hyperparameters
batch_size = 16
learning_rate = 1e-5


# In[5]:


# Load and preprocess the dataset
def load_dataset(root_dir):
    data = []
    labels = []
    for label in ['pos', 'neg']:
        label_dir = os.path.join(root_dir, label)
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                data.append(text)
                rating = int(filename.split('_')[-1].split('.')[0])
                normalized_rating = (rating - 1) / 9.0
                labels.append(normalized_rating)
    return data, labels

train_dir = 'aclImdb/train'
test_dir = 'aclImdb/test'
tokenizer = get_tokenizer('basic_english')

train_data, train_labels = load_dataset(train_dir)
test_data, test_labels = load_dataset(test_dir)


# In[6]:


def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_dataset = MovieReviewDataset(train_data, train_labels, tokenizer)
test_dataset = MovieReviewDataset(test_data, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)


# In[ ]:


# Initialize the model
model = DistilBertModelForRegression(1).to(device)

# Unfreeze the DistilBERT layers
for param in model.distilbert.parameters():
    param.requires_grad = True

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Set up directories
print("Creating necessary directories...")
model_name = 'bert1'
plot_dir = os.path.join(model_name, 'plots')
checkpoint_dir = os.path.join(model_name, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
info_dir = os.path.join(model_name, 'info')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(info_dir, exist_ok=True)
loss_plot_path = os.path.join(plot_dir, 'training_validation_loss.png')
checkpoint_increment = 20

# Load saved state dictionaries if they exist
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    start_epoch = 0
    file_path = os.path.join(info_dir, 'architecture.txt')
    model_info = str(model)
    with open(file_path, 'w') as file:
        file.write(model_info)
    print(f"Model architecture saved to {file_path}")
    train_losses = []
    test_losses = []
    
print('Starting the training...')
try:
    epoch = start_epoch
    while True:
        model.train()
        train_loss = 0.0

        for input_ids, attention_masks, targets in train_loader:
            input_ids, attention_masks, targets = input_ids.to(device), attention_masks.to(device), targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_masks).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for input_ids, attention_masks, targets in test_loader:
                input_ids, attention_masks, targets = input_ids.to(device), attention_masks.to(device), targets.to(device).float()
                outputs = model(input_ids, attention_masks).squeeze()
                loss = criterion(outputs, targets)
                test_loss += loss.item() * input_ids.size(0)


        test_loss /= len(test_dataset)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}], Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses
        }, checkpoint_path)
        
        if (epoch + 1) % checkpoint_increment == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses
            }, os.path.join(checkpoint_dir, f'checkpoint{epoch + 1}.pt'))

        # Plotting and saving the figure
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()

        epoch += 1

except KeyboardInterrupt:
    print('Training has been manually interrupted.')