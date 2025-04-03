import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from sklearn.model_selection import KFold
import sys
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import numpy as np

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
K_FOLDS = 5  # Number of folds for cross-validation
USE_KFOLD = True  # Toggle k-fold cross-validation on/off

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

# Log cross-validation experiment results
def log_cv_experiment(fold_accuracies, avg_accuracy, std_accuracy, model):
    dir_path = f'{LOG_DIR}/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_kfold_cv'
    os.makedirs(dir_path, exist_ok=True)

    with open(f'{dir_path}/fold_accuracies.txt', 'w') as file:
        for i, acc in enumerate(fold_accuracies):
            file.write(f"Fold {i+1}: {acc:.2f}%\n")
        file.write(f"\nAverage Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    
    with open(f'{dir_path}/model_architecture.txt', 'w') as file:
        file.write(str(model))

    hyperparameters = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'K_FOLDS': K_FOLDS
    }
    with open(f'{dir_path}/hyperparameters.json', 'w') as file:
        json.dump(hyperparameters, file, indent=4)
    
    # Plot fold accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, K_FOLDS + 1), fold_accuracies)
    plt.axhline(y=avg_accuracy, color='r', linestyle='-', label=f'Mean: {avg_accuracy:.2f}%')
    plt.axhline(y=avg_accuracy + std_accuracy, color='g', linestyle='--', label=f'+1 STD: {avg_accuracy + std_accuracy:.2f}%')
    plt.axhline(y=avg_accuracy - std_accuracy, color='g', linestyle='--', label=f'-1 STD: {avg_accuracy - std_accuracy:.2f}%')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title('K-Fold Cross-Validation Results')
    plt.legend()
    plt.savefig(f'{dir_path}/fold_accuracies.png')
    
    # Return the path where results are saved
    return dir_path

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

def ensemble_predict(models, dataloader, output_dim):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            preds_sum = torch.zeros(x.size(0), output_dim)
            for m in models:
                preds_sum += m(x)
            avg_preds = preds_sum / len(models)
            all_preds.append(avg_preds)
            all_targets.append(y)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    predicted_classes = torch.argmax(all_preds, dim=1)
    target_classes = torch.argmax(all_targets, dim=1)
    correct = torch.sum(predicted_classes == target_classes).item()
    accuracy = correct / all_preds.size(0) * 100
    return accuracy

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
    X = torch.tensor(df.iloc[:, 1:-output_dim].values, dtype=torch.float32)  # Features
    y = torch.tensor(df.iloc[:, -output_dim:].values, dtype=torch.float32)  # Labels
    dataset = TensorDataset(X, y)

    if USE_KFOLD:
        # K-fold cross-validation
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_models = []
        
        print("\n" + "="*50)
        print(f"STARTING {K_FOLDS}-FOLD CROSS VALIDATION")
        print("="*50 + "\n")
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            print(f"\nFOLD {fold+1}/{K_FOLDS}")
            print("-" * 30)
            
            # Create train and validation splits
            train_size = int(0.8 * len(train_ids))
            val_ids = train_ids[train_size:]
            train_ids = train_ids[:train_size]
            
            train_subsampler = Subset(dataset, train_ids)
            val_subsampler = Subset(dataset, val_ids)
            test_subsampler = Subset(dataset, test_ids)
            
            train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
            test_loader = DataLoader(test_subsampler, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
            
            # Initialize model and optimizer for this fold
            model = SoftmaxRegression(input_dim, output_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            loss_fn = nn.CrossEntropyLoss()
            
            # Train the model
            train_loss_history = []
            val_loss_history = []
            accuracy_history = []
            
            for epoch in range(NUM_EPOCHS):
                train_loss = train_loop(train_loader, model, loss_fn, optimizer)
                val_loss, accuracy = validate(val_loader, model, loss_fn)
                
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                accuracy_history.append(accuracy)
                
                if epoch % 10 == 0:
                    print(f'Epoch: [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            # Evaluate on test set for this fold
            _, test_accuracy = validate(test_loader, model, loss_fn)
            print(f"FOLD {fold+1} Test Accuracy: {test_accuracy:.2f}%")
            fold_accuracies.append(test_accuracy)
            fold_models.append(model)
        
        # Compute mean and standard deviation of fold accuracies
        avg_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        
        print("\n" + "="*50)
        print("CROSS VALIDATION RESULTS")
        print("="*50)
        for i, acc in enumerate(fold_accuracies):
            print(f"Fold {i+1}: {acc:.2f}%")
        print("-"*30)
        print(f"Average Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")

        print("="*50 + "\n")
        
        # Log cross-validation results
        result_path = log_cv_experiment(fold_accuracies, avg_accuracy, std_accuracy, model)
        print(f"Cross-validation results saved to: {result_path}")
    
    else:
        # Original split code without k-fold
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
            ensemble_accuracy = ensemble_predict(ensemble_models, val_loader, output_dim)
            print(f"Ensemble validation accuracy: {ensemble_accuracy:.2f}%")
            
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
