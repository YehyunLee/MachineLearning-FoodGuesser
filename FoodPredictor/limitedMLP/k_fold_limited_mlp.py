import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from preprocess import preprocess
from datetime import datetime
import json
import os

torch.manual_seed(42)  # for reproducibility

DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'
LOG_DIR = '../limitedMLP/experiment_logs'
NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-3

model = nn.Sequential(
    nn.Linear(19, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 3)  # nn.CrossEntropyLoss applies softmax internally during loss calculation
)


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

    
def train(train_loader, val_loader, model, loss_fn, optimizer):
    train_loss_history = []
    val_loss_history = []
    accuracy_history = []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        train_loss_history.append(train_loss)

        val_running_loss = 0
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = loss_fn(pred, y)
                val_running_loss += loss.item()

                num_correct += torch.sum(torch.argmax(pred, axis=1) == torch.argmax(y, axis=1))
                num_samples += x.size(0)

        val_loss_history.append(val_running_loss / len(val_loader))
        accuracy_history.append(num_correct.item() / num_samples * 100)
        print(f'Epoch: [{epoch + 1}/{NUM_EPOCHS}], train_loss: {train_loss}, val_loss: {val_running_loss / len(val_loader)}, accuracy: {num_correct.item() / num_samples * 100}%')

    log_experiment(train_loss_history, val_loss_history, accuracy_history, model)


def log_experiment(train_loss_history, val_loss_history, accuracy_history, model):
    dir_path = f'{LOG_DIR}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(dir_path, exist_ok=True)

    with open(f'{dir_path}/train_loss_history.txt', 'w') as file:
        file.write('\n'.join([str(loss) for loss in train_loss_history]))
    
    with open(f'{dir_path}/val_loss_history.txt', 'w') as file:
        file.write('\n'.join([str(loss) for loss in val_loss_history]))

    with open(f'{dir_path}/accuracy_history.txt', 'w') as file:
        file.write('\n'.join([str(accuracy) for accuracy in accuracy_history]))

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


if __name__ == '__main__':
    df = preprocess(DATASET_PATH)

    dataset = TensorDataset(
        torch.tensor(df.iloc[:,:-3].to_numpy(), dtype=torch.float32),
        torch.tensor(df.iloc[:,-3:].to_numpy(), dtype=torch.float32)
    )

    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train(train_loader, val_loader, model, loss_fn, optimizer)


    # --- New Code: 5-Fold Cross-Validation ---
    from torch.utils.data import Subset
    from sklearn.model_selection import KFold

    def k_fold_cv(dataset, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_accuracies = []
        indices = list(range(len(dataset)))
        fold = 1
        for train_idx, val_idx in kf.split(indices):
            print(f"\nFold {fold}:")
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader_cv = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader_cv = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
            
            # Re-create the model for each fold (using the same architecture)
            model_cv = nn.Sequential(
                nn.Linear(19, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 3)
            )
            
            loss_fn_cv = nn.CrossEntropyLoss()
            optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            # Train the model on the current fold
            train(train_loader_cv, val_loader_cv, model_cv, loss_fn_cv, optimizer_cv)
            
            # Evaluate the model on the validation subset for this fold
            model_cv.eval()
            num_correct = 0
            num_samples = 0
            with torch.no_grad():
                for x, y in val_loader_cv:
                    pred = model_cv(x)
                    num_correct += torch.sum(torch.argmax(pred, axis=1) == torch.argmax(y, axis=1))
                    num_samples += x.size(0)
            fold_accuracy = num_correct.item() / num_samples * 100
            print(f'Fold {fold} Accuracy: {fold_accuracy}%')
            fold_accuracies.append(fold_accuracy)
            fold += 1
        
        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print(f'\nMean 5-Fold CV Accuracy: {mean_accuracy:.2f}%')
        return fold_accuracies

    print("\nPerforming 5-Fold Cross-Validation on the entire dataset:")
    k_fold_cv(dataset, k=5)
