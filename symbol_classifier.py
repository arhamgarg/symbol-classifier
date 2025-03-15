from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, has_labels=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = self.root_dir / img_name
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('L', (32, 32), 0)

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = self.annotations.iloc[index, 1]
            return image, torch.tensor(label, dtype=torch.long)
        return image


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomAutocontrast(p=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3)
])

train_data = CustomDataset(csv_file='content/dataset/train.csv',
                           root_dir='content/dataset/', transform=transform, has_labels=True)
test_data = CustomDataset(csv_file='content/dataset/test.csv',
                          root_dir='content/dataset/', transform=transform, has_labels=False)

label_encoder = LabelEncoder()
train_data.annotations['label'] = label_encoder.fit_transform(train_data.annotations['label'])

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, drop_last=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        num_classes = len(pd.unique(train_data.annotations['label']))
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train_model(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0

    for images, labels in loader:
        if images.size(0) != labels.size(0):
            print(f"Batch size mismatch: images={images.size(0)}, labels={labels.size(0)}")
            continue

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    return epoch_loss / len(loader), accuracy


def evaluate_model(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, tuple):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
            else:
                images = batch.to(device)
                outputs = model(images)

    if total_samples > 0:
        accuracy = (correct_predictions / total_samples) * 100
        return test_loss / len(loader), accuracy

    return None, None


num_epochs = 20
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
    print(f"After train_model: loss={train_loss:.4f}, acc={train_acc:.2f}%")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"After evaluate_model: loss={test_loss}, acc={test_acc}")

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Test Loss: {test_loss if test_loss is not None else 'N/A'}, "
          f"Test Acc: {test_acc if test_acc is not None else 'N/A'}%")

torch.save(model.state_dict(), "symbol_model.pth")
print("Model saved successfully.")


def generate_submission(model, test_loader, test_csv_path, output_file="submission.csv"):
    model.eval()
    predictions = []
    example_ids = pd.read_csv(test_csv_path)['example_id'].tolist()

    with torch.no_grad():
        for batch in test_loader:
            images = batch.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    original_labels = label_encoder.inverse_transform(predictions)

    submission_df = pd.DataFrame({
        "example_id": example_ids[:len(predictions)],
        "label": original_labels
    })

    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved as {output_file}")


generate_submission(model, test_loader, "content/dataset/test.csv")
