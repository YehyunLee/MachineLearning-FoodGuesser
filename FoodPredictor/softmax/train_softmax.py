import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import sys
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

sys.path.append('../utils')
from preprocess import preprocess

torch.manual_seed(42)  # for reproducibility

DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'
LOG_DIR = '../softmax/experiment_logs'
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_BAGS = 5  # number of bootstrap models
USE_BAGGING = True  # Toggle bagging on/off

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
    num_samples = 0
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        num_samples += x.size(0)
        running_loss += loss.item() * x.size(0)

    return running_loss / num_samples

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
            val_loss += loss.item() * x.size(0)

            num_correct += torch.sum(torch.argmax(pred, axis=1) == torch.argmax(y, axis=1))
            num_samples += x.size(0)

    accuracy = num_correct.item() / num_samples * 100
    return val_loss / num_samples, accuracy

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

def log_ensemble_experiment(ensemble_accuracy, ensemble_models):
    # Create a timestamped directory for ensemble logs
    dir_path = f'{LOG_DIR}/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_ensemble'
    os.makedirs(dir_path, exist_ok=True)
    
    with open(f'{dir_path}/ensemble_accuracy.txt', 'w') as file:
        file.write(str(ensemble_accuracy))
    
    with open(f'{dir_path}/model_architecture.txt', 'w') as file:
        # Log each model's architecture
        file.write("\n".join([str(model) for model in ensemble_models]))
    
    hyperparameters = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'N_BAGS': N_BAGS
    }
    with open(f'{dir_path}/hyperparameters.json', 'w') as file:
        json.dump(hyperparameters, file, indent=4)
 

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

def ensemble_predict(models, dataloader, output_dim, loss_fn):
    all_preds = []
    all_targets = []
    running_loss = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            preds_sum = torch.zeros(x.size(0), output_dim)
            for m in models:
                preds_sum += m(x)
                
            avg_preds = preds_sum / len(models)
            running_loss += loss_fn(avg_preds, y).item() * x.size(0)
            total_samples += x.size(0)
            all_preds.append(avg_preds)
            all_targets.append(y)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    predicted_classes = torch.argmax(all_preds, dim=1)
    target_classes = torch.argmax(all_targets, dim=1)
    correct = torch.sum(predicted_classes == target_classes).item()
    accuracy = correct / all_preds.size(0) * 100
    loss = running_loss / total_samples
    return accuracy, loss

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

    # Initialize loss function
    loss_fn = nn.CrossEntropyLoss()
    
    if USE_BAGGING:
        # Bagging: train ensemble of models using bootstrapped samples
        ensemble_models = []
        for bag in range(N_BAGS):
            print(f"Training bag {bag+1}/{N_BAGS}")
            # Create bootstrap sample indices
            indices = torch.randint(0, len(train_dataset), (len(train_dataset),))
            bootstrap_subset = torch.utils.data.Subset(train_dataset, indices.tolist())
            bootstrap_loader = DataLoader(bootstrap_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            model = SoftmaxRegression(input_dim, output_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            for epoch in range(NUM_EPOCHS):
                train_loss = train_loop(bootstrap_loader, model, loss_fn, optimizer)
                if epoch % 10 == 0:
                    print(f"Bag {bag+1} Epoch {epoch}: Loss {train_loss:.4f}")
            ensemble_models.append(model)
        
        # Evaluate ensemble on validation data
        ensemble_accuracy, ensemble_loss = ensemble_predict(ensemble_models, val_loader, output_dim, loss_fn)
        print(f"Ensemble validation accuracy: {ensemble_accuracy:.2f}%")
        print(f"Ensemble validation loss: {ensemble_loss}")
        
        # Log the ensemble experiment similar to mlp train.py logs
        log_ensemble_experiment(ensemble_accuracy, ensemble_models)
    else:
        # Train a single model without bagging
        model = SoftmaxRegression(input_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        train(train_loader, val_loader, model, loss_fn, optimizer)

        # Evaluate the single model on validation data
        val_loss, val_accuracy = validate(val_loader, model, loss_fn)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
