import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CSV files
styles_df = pd.read_csv('data/fashion-dataset/styles.csv', encoding='utf-8')
images_df = pd.read_csv('data/fashion-dataset/images.csv', encoding='utf-8')

# Debugging print statements
print(f"\U0001F4CC styles.csv columns: {styles_df.columns.tolist()}")
print(f"\U0001F4CC images.csv columns: {images_df.columns.tolist()}")
print(f"\U0001F6E0 First few rows of styles.csv:\n{styles_df.head()}")
print(f"\U0001F6E0 First few rows of images.csv:\n{images_df.head()}")

# Ensure filename/id columns match correctly
styles_df['id'] = styles_df['id'].astype(str) + '.jpg'

# Merge DataFrames
merged_df = images_df.merge(styles_df, left_on='filename', right_on='id', how='inner')
print(f"\u2705 After merging: {len(merged_df)} records found")

# Filter valid image files
image_folder = 'data/fashion-dataset/images'
merged_df = merged_df[merged_df['filename'].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))]
print(f"Valid dataset size: {len(merged_df)}")

if len(merged_df) == 0:
    raise ValueError("No valid images found. Please check the dataset paths and files.")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset class
class FashionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = {label: idx for idx, label in enumerate(dataframe['articleType'].unique())}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(image_folder, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[row['articleType']]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Create DataLoader
dataset = FashionDataset(merged_df, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load Pretrained CNN Model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(merged_df['articleType'].unique()))
model = model.to(device)

# Define Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
train_losses = []
accuracy_list = []
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}] started")
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i + 1) % 10 == 0:
            print(f"Batch [{i+1}/{len(data_loader)}] - Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(data_loader)
    train_losses.append(epoch_loss)
    epoch_accuracy = 100 * correct / total
    accuracy_list.append(epoch_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save the trained model
os.makedirs("models", exist_ok=True)
model_path = "models/fashion_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plot Training Loss & Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), accuracy_list, label='Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Over Epochs')
plt.legend()
plt.show()

print("Training completed.")
