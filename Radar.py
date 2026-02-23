#!/usr/bin/env python
# coding: utf-8

# <img src="./figs/IOAI-Logo.png" alt="IOAI Logo" width="200" height="auto">
# 
# [IOAI 2025 (Beijing, China), Individual Contest](https://ioai-official.org/china-2025)
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IOAI-official/IOAI-2025/blob/main/Individual-Contest/Radar/Radar.ipynb)

# # Radar
# 
# ## 1. Problem Description 
# 
# Radar is a key technology in wireless communication, with widespread applications such as self-driving cars. It typically involves an antenna that transmits specific signals and receives their reflections from objects in the environment. By processing these signals, the system determines the angular direction, distance, and velocity of target objects.
# 
# In real-world applications, radar signal processing is challenging due to noise and reflections from non-target objects in the surroundings. For example, when attempting to detect pedestrians, the radar may also receive reflections from trees or other background objects, which can degrade accuracy. Your task is to use AI to analyze the signals received by the radar and identify the presence of a human at each position.
# 
# In this task, we provide an **indoor radar experiment dataset**, and your objective is to develop a model that performs **radar semantic segmentation**. 
# 
# 
# ## 2. Dataset
# 
# To measure objects surrounding a radar, the following key parameters are used:
# 
# - **Range**: The straight-line distance between the radar and an object.
# - **Azimuth**: The horizontal angle (left to right) between the radar and the object.
# - **Elevation**: The vertical angle (up or down) of the object relative to the radar.
# - **Velocity**: The speed at which the object is moving toward or away from the radar.
# 
# <img src="./figs/Radar Fig 1.png" width="300">
# 
# 
# The radar data is processed into multiple **heatmaps**, each encoding the **received signal strength** at various positions and directions.
# 
# - **Static heatmaps** emphasize reflections from **stationary** objects.
# - **Dynamic heatmaps** highlight changes caused by **moving** objects.
# 
# When no object is present at a specific location, the signal consists mostly of background noise and appears weak. In contrast, reflections from an object increase signal intensity, enabling detection of the object.
# 
# For example, the **static range-azimuth heatmap** represents signal strength across different distances (**range**) and horizontal angles (**azimuth**), mainly reflected by stationary objects.
# 
# Each sample in the dataset is stored in a `.mat.pt` file as a tensor of shape $7 \times 50 \times 181$, where:
# 
# - 7 is the number of maps (6 heatmaps + 1 semantic label map),
# - 50 represents range bins (distance),
# - 181 represents angular or velocity bins, covering angles from \-90° to \+90° in either the horizontal or vertical plane. You can assume that the velocity bins are also remapped from \-90° to \+90° for visualization consistency.
# - each heatmap intensity value is normalized to [0, 1], representing received signal strength.
# 
# The 6 heatmaps are structured as follows:
# 
# - **Index 0**: Static range-azimuth heatmap
# - **Index 1**: Dynamic range-azimuth heatmap
# - **Index 2**: Static range-elevation heatmap
# - **Index 3**: Dynamic range-elevation heatmap
# - **Index 4**: Static range-velocity heatmap
# - **Index 5**: Dynamic range-velocity heatmap
# 
# All values in heatmaps are **normalized**, so no unit conversion is required.
# 
# The **map at Index 6** is the semantic label map, stored in range-azimuth format. 
# 
# - **-1**: Background (no target)
# - **0**: Suitcase
# - **1**: Chair
# - **2**: Human
# - **3**: Wall
# 
# This is the visualization of 1.mat.pt in training_set:
# 
# <img src="./figs/Radar Fig 2.png" width="675">
# 
# Here is part of a sample from the dataset:
# 
# <img src="./figs/Radar Fig 3.png" width="675">
# 
# 
# Data scale: 1800 samples in the training set, 500 samples in the validation set, and 500 samples in the test set.
# 
# ## 3\. Task
# 
# Your task is to develop a model that takes the **first six heatmaps** (indices 0 to 5) as input, and predicts the **semantic label map** (index 6) as the output. The goal is to accurately identify what the target is(-1 to 3) at each location in the radar’s field of view.
# 
# 1. **Input**: A tensor of shape $6 \times 50 \times 181$, representing six radar heatmaps.
# 2. **Output**: A tensor of shape $50 \times 181$, representing the target semantic label map.
# 
# 
# ## 4\. Submission 
# 
# Please submit a file named `submission.ipynb`. The output is a zip file named "submission.zip", which contains two tables `submission_val.csv` and `submission_test.csv` corresponding to the prediction results of the validation set and the test set respectively.
# 
# **Note:** The output table should have a header, the data in the table is not the actual solved data, it is only used as an example of the submission format.
# 
# | filename | pixel_0 | pixel_1 | ... | pixel_9049 |
# | :------: | :-----: | ------- | --- | ---------- |
# | 1.mat.pt |   -1    | -1      | ... | -1         |
# |   ...    |   ...   | ...     | ... | ...        |
# 
# ## 5\. Score
# 
# The score is based on the **accuracy of label recognition**. Correctly identifying target points is weighted more heavily than correctly identifying background points. 
# 
# ### Scoring Criteria: 
# 
# * Each correctly identified **background pixel** earns **1 point**. 
# 
# * Each correctly identified **non-background pixel** earns **50 points**. 
# 
# * The final score is normalized to a **0-1 point** by comparing it to the maximum possible score. 
# 
# ### Formula：
# $$
# Score = \frac{|C_{0,correct}| \times 1 + |C_{1,correct}| \times bonus}{|C_0| \times 1 + |C_1| \times bonus}
# $$
# where:
# 
# $$
# \begin{aligned}
# I &= \{1, 2, \dots, 50\times 181\}\\
# C_0 &= \{i \in I \mid y_i = -1\}\\
# C_1 &= \{i \in I \mid y_i \neq -1\}\\
# C_{0,correct} &= \{i \in C_0 \mid p_i = y_i\}\\
# C_{1,correct} &= \{i \in C_1 \mid p_i = y_i\}\\
# \end{aligned}
# $$
# 
# 
# ### Example
# 
# For a $3\times3$ heatmap, assume the Ground Truth is:
# 
# $$
# \begin{bmatrix}
# -1 & -1 & -1 \\
# 1 & 2 & 3 \\
# -1 & -1 & -1
# \end{bmatrix}
# $$
# 
# The intenteded result is:
# 
# $$
# \begin{bmatrix}
# -1 & 1 & -1 \\
# -1 & 2 & -1 \\
# -1 & 3 & -1
# \end{bmatrix}
# $$
# 
# Then there are four correctly identified `-1` and one correctly identified `2`. Your score is 4 + 50 = 54 points. The maximum possible score is 6 + 50 * 3 = 156, that is, the score for six background pixels and three non-background pixels. Your normalized score is 54 / 156 = 0.346.
# 
# $$
# Score = \frac{4 \times 1 + 1 \times 50}{6 \times 1 + 3 \times 50}=0.346
# $$
# 
# ## 6. Baseline and Training Set
# 
# - Below you can find the baseline solution.
# - The dataset is in `training_set` folder.
# - The highest score by the Scientific Committee for this task is 0.90 in Leaderboard B, this score is used for score unification.
# - The baseline score by the Scientific Committee for this task is 0.67 in Leaderboard B, this score is used for score unification.

# ### Data Loading

# In[ ]:


import random
import numpy as np
import torch

seed = 42
NUM_WORKERS = 0
random.seed(seed)                  # Python built-in random
np.random.seed(seed)               # NumPy
torch.manual_seed(seed)            # PyTorch (CPU)
torch.cuda.manual_seed(seed)       # PyTorch (single GPU)
torch.cuda.manual_seed_all(seed)   # PyTorch (all GPUs)

# Ensures deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[ ]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.file_names = [os.path.basename(path) for path in file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=True)

        images = data[:6]  
        labels = data[6]           

        images = images.float()  
        labels = labels.long()   
        labels = labels + 1

        if self.transform:
            images = self.transform(images)
            labels = self.transform(labels)

        return images, labels, self.file_names[idx]

class CustomDataset_test(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.file_names = [os.path.basename(path) for path in file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=True)

        images = data[:6]        

        images = images.float()  

        if self.transform:
            images = self.transform(images)

        return images, self.file_names[idx]

def generate_file_paths(base_path):
    file_paths = []
    for frame in os.listdir(base_path):
        frame_path = os.path.join(base_path, frame)
        if frame_path.endswith('.mat.pt'):
            file_paths.append(frame_path)
    return [path for path in file_paths if os.path.exists(path)]

def load_data(base_path, batch_size=4, num_workers=NUM_WORKERS, test_size=0.2):
    file_paths = generate_file_paths(base_path)

    train_paths, test_paths = train_test_split(file_paths, test_size=test_size, random_state=42)

    train_dataset = CustomDataset(file_paths=train_paths)
    test_dataset = CustomDataset(file_paths=test_paths)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        drop_last=True
    )

    return train_loader, test_loader


# ### Model Definition and Training

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=5, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)  
        return x

def train(model, train_loader, test_loader, optimizer, criterion, num_epochs=100):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        print(f"Epoch {epoch+1}/{num_epochs} - Training...")
        for images, labels, _ in train_loader:
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            outputs = model(images)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)  # [B, C, H*W]
            labels = labels.view(labels.size(0), -1)  # [B, H*W]
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.cuda() if torch.cuda.is_available() else images
                labels = labels.cuda() if torch.cuda.is_available() else labels

                outputs = model(images)
                outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
                labels = labels.view(labels.size(0), -1)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)

        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses
TRAIN_PATH = "./"
# The training set is deployed automatically in the testing machine. 
# You notebook can access the TRAIN_PATH even if you do not mount it along with notebook.
data_path = TRAIN_PATH + 'training_set'

train_loader, test_loader = load_data(
    base_path=data_path,
    batch_size=4,  
    num_workers=NUM_WORKERS, #  !!!!!!!! they were 2
    test_size=0.2
)

model = MyModel()
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005) 

train_losses, val_losses = train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=100
)


# ### Generate CSV for Submission

# In[ ]:


# Run inference on validation set and testing set
from torch.utils.data import DataLoader
import pandas as pd

def run_inference(model, data_loader):
    """Run inference and return predictions with filenames"""
    model.eval()
    predictions = []
    filenames = []

    with torch.no_grad():
        for images, file_names in data_loader:
            images = images.cuda() if torch.cuda.is_available() else images

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Convert predictions back to original label range [-1, 3]
            preds = preds - 1

            # Flatten predictions for each sample
            for i, pred in enumerate(preds):
                predictions.append(pred.cpu().numpy().flatten())
                filenames.append(file_names[i])

    return predictions, filenames

#DATA_PATH is the secret environment variable to point the address of the validation set and test set on the testing machine. 
#You cannot access this address locally.
if os.environ.get('DATA_PATH'):
    DATA_PATH = os.environ.get("DATA_PATH") + "/" 
else:
    DATA_PATH = "Solution/"  # Fallback for local testing
# Load validation set
val_paths = generate_file_paths(DATA_PATH + 'validation_set')
val_dataset = CustomDataset_test(file_paths=val_paths)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# Load testing set
test_paths = generate_file_paths(DATA_PATH + 'test_set')
test_dataset = CustomDataset_test(file_paths=test_paths)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS

)

# Run inference on validation set
print("Running inference on validation set...")
val_predictions, val_filenames = run_inference(model, val_loader)

# Save validation results to CSV
val_results = []
for filename, pred in zip(val_filenames, val_predictions):
    # Create a row with filename and flattened predictions
    row = {'filename': filename}
    for i, p in enumerate(pred):
        row[f'pixel_{i}'] = p
    val_results.append(row)

val_df = pd.DataFrame(val_results)
val_df.to_csv('submission_val.csv', index=False)
print(f"Validation results saved to output_validation.csv with shape: {val_df.shape}")

# Run inference on testing set
print("Running inference on testing set...")
test_predictions, test_filenames = run_inference(model, test_loader)

# Save testing results to CSV
test_results = []
for filename, pred in zip(test_filenames, test_predictions):
    # Create a row with filename and flattened predictions
    row = {'filename': filename}
    for i, p in enumerate(pred):
        row[f'pixel_{i}'] = p
    test_results.append(row)

test_df = pd.DataFrame(test_results)
test_df.to_csv('submission_test.csv', index=False)
print(f"Testing results saved to output_testing.csv with shape: {test_df.shape}")

print("\nInference completed! Results saved to:")
print("- submission_val.csv (for validation set leaderboard)")
print("- submission_test.csv (for testing set leaderboard)")


# ### Create .zip File

# In[ ]:


import zipfile
import os

# Define the files to zip and the zip file name.
files_to_zip = ['submission_val.csv', 'submission_test.csv']
zip_filename = 'submission.zip'

# Create a zip file
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in files_to_zip:
        # Add the file to the zip fil
        zipf.write(file, os.path.basename(file))

print(f'{zip_filename} Created successfully!')


# %%
from IPython import get_ipython
get_ipython().run_line_magic('run', 'metrics.py')

# %%
