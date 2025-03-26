import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from preprocess import preprocess
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

torch.manual_seed(42)  # for reproducibility

DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'
LOG_DIR = '../softmax/experiment_logs'
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Define the Softmax Regression model
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)  # Raw logits; nn.CrossEntropyLoss applies softmax internally

# Training loop
def train_loop(train_loader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Validation loop
def validate(val_loader, model, loss_fn):
    model.eval()
    val_loss = 0
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss += loss.item()

            num_correct += torch.sum(torch.argmax(pred, axis=1) == torch.argmax(y, axis=1))
            num_samples += x.size(0)

    accuracy = num_correct.item() / num_samples * 100
    return val_loss / len(val_loader), accuracy

# Logging experiment results
def log_experiment(train_loss_history, val_loss_history, accuracy_history, model):
    dir_path = f'{LOG_DIR}/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    os.makedirs(dir_path, exist_ok=True)

    with open(f'{dir_path}/train_loss_history.txt', 'w') as file:
        file.write('\n'.join([str(loss) for loss in train_loss_history]))
    
    with open(f'{dir_path}/val_loss_history.txt', 'w') as file:
        file.write('\n'.join([str(loss) for loss in val_loss_history]))

    with open(f'{dir_path}/accuracy_history.txt', 'w') as file:
        file.write('\n'.join([str(acc) for acc in accuracy_history]))

    with open(f'{dir_path}/model_architecture.txt', 'w') as file:
        file.write(str(model))

    hyperparameters = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY
    }
    with open(f'{dir_path}/hyperparameters.json', 'w') as file:
        json.dump(hyperparameters, file, indent=4)

    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{dir_path}/loss.png')

# Main training function
def train(train_loader, val_loader, model, loss_fn, optimizer):
    train_loss_history = []
    val_loss_history = []
    accuracy_history = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        val_loss, accuracy = validate(val_loader, model, loss_fn)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        accuracy_history.append(accuracy)

        print(f'Epoch: [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    log_experiment(train_loss_history, val_loss_history, accuracy_history, model)

if __name__ == '__main__':
    # For Softmax training, use bag-of-words from Q5 and Q6 (and Label)
    df = preprocess(DATASET_PATH, normalize_and_onehot=True, mode="softmax")

    # Check for NaN values in the preprocessed data
    if df.isnull().values.any():
        raise ValueError("Preprocessed data contains NaN values. Please check the preprocessing pipeline.")

    # Ensure the Label column is present and correctly handled
    label_col = [col for col in df.columns if col.startswith("Label_")]
    if not label_col:
        raise ValueError("Label column is missing in the preprocessed data.")

    # Prepare dataset: exclude 'id' (first) and use last label columns for targets
    input_dim = df.shape[1] - len(label_col) - 1  # Exclude "id" and label columns
    output_dim = len(label_col)  # Number of classes
    dataset = TensorDataset(
        torch.tensor(df.iloc[:, 1:-output_dim].to_numpy(), dtype=torch.float32),  # Exclude "id" and label columns
        torch.tensor(df.iloc[:, -output_dim:].to_numpy(), dtype=torch.float32)
    )

    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # Initialize model, loss function, and optimizer
    model = SoftmaxRegression(input_dim, output_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train the model
    train(train_loader, val_loader, model, loss_fn, optimizer)
