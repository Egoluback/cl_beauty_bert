import os
import csv
import json
import time
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import ModernBertForSequenceClassification, ModernBertTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Constants and Hyperparameters
BASELINE_EPOCH = -1
BATCH_SIZE = 16
NUM_EPOCHS = 3         # number of curriculum outer epochs
SAMPLE_TRAIN_EPOCHS = 2  # mini-epochs when training on each sample to compute improvement
THRESHOLD_RATIO = 0.25    # select top fraction of samples
MAX_LENGTH = 256
LR = 2e-5
WEIGHT_SAVE_MODULUS = 1  # save weights every epoch
OUTPUT_DIR = "./modernbert_output"
CSV_HEADER = ["Epoch", "Learning_Rate", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"]

# Create output directories if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
weights_folder = os.path.join(OUTPUT_DIR, 'weights')
os.makedirs(weights_folder, exist_ok=True)
results_folder = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(results_folder, exist_ok=True)
results_file = os.path.join(results_folder, 'results.csv')
best_loss_file = os.path.join(results_folder, 'best_loss_weights.pth')
best_acc_file = os.path.join(results_folder, 'best_acc_weights.pth')
best_text_file = os.path.join(results_folder, 'best_epochs.txt')

# Utility: IMDB dataset torch wrapper
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                  max_length=self.max_length, return_tensors="pt")
        # Squeeze to remove batch dim and create item structure.
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Utility: Evaluate model on dataloader
def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += input_ids.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# Utility: Training epoch on a given dataloader
def train_epoch(epoch, model, dataloader, loss_fn, optimizer, device, print_modulus=10):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += input_ids.size(0)
        if (i+1) % print_modulus == 0:
            print(f"[Epoch {epoch}] Batch {i+1}/{len(dataloader)}")

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# Curriculum learning: Compute loss improvement for each sample
def train_on_samples(model, dataloader, loss_fn, optimizer_lr, device):
    results = []  # List of tuples: (improvement_ratio, sample index)
    # Iterate over each sample (batch size == 1)
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Sample Training"):
        # Make a deep copy of the model for per-sample training
        sample_model = copy.deepcopy(model)
        sample_model.to(device)
        sample_model.eval()

        # Compute initial loss on the single sample
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = sample_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            initial_loss = loss_fn(logits, labels).item()

        # Create a fresh optimizer for the copied model
        sample_optimizer = Adam(sample_model.parameters(), lr=optimizer_lr)

        # Training on the single sample for several mini-epochs
        for _ in range(SAMPLE_TRAIN_EPOCHS):
            sample_model.train()
            sample_optimizer.zero_grad()
            outputs = sample_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            sample_optimizer.step()

        # Evaluate loss after fine-tuning on this sample
        sample_model.eval()
        with torch.no_grad():
            outputs = sample_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            final_loss = loss_fn(logits, labels).item()

        # Compute improvement ratio. Higher ratio indicate better potential to learn from this sample.
        improvement_ratio = (initial_loss - final_loss) / (initial_loss + 1e-10)
        results.append((improvement_ratio, idx))
        # Optionally, log sample id and improvement value
        print(f"Sample index: {idx}, Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}, Improvement: {improvement_ratio:.4f}")

    return results

# Select the best samples based on improvement ratio
def choose_best_samples(diffs, epoch, threshold=THRESHOLD_RATIO):
    diffs.sort(key=lambda x: x[0], reverse=True)
    top_count = max(1, int(len(diffs) * threshold))
    best_samples_indices = [idx for _, idx in diffs[:top_count]]
    # Save best sample indices for the epoch into a file for record
    file_path = os.path.join(OUTPUT_DIR, f'best_samples_epoch_{epoch}.txt')
    with open(file_path, 'w') as f:
        for imp, idx in diffs:
            f.write(f"Index: {idx}\tImprovement: {imp:.4f}\n")
    return best_samples_indices

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load tokenizer and model
    tokenizer = ModernBertTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2)
    model.to(device)

    # Load IMDB dataset
    dataset = load_dataset("wics/strategy-qa")
    # For simplicity, use the 'train' split for training and a small portion as validation.
    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]
    full_train_dataset = IMDBDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    
    # Create a small validation set: use the 'test' split from imdb as validation
    val_texts = dataset["test"]["text"]
    val_labels = dataset["test"]["label"]
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length=MAX_LENGTH)
    
    # DataLoader for baseline/validation evaluation. Standard batch size for training epochs.
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Prepare CSV file for recording
    if not os.path.isfile(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    best_eval_acc = 0.0
    best_eval_acc_epoch = -1
    best_eval_loss = float("inf")
    best_eval_loss_epoch = -1

    # Outer loop: Curriculum learning epochs
    for total_epoch in range(BASELINE_EPOCH+1, NUM_EPOCHS):
        print("="*50)
        print(f"Curriculum Epoch: {total_epoch+1}")
        print("="*50)

        # Step 1: Evaluate each sample improvement on the full training set
        # Use DataLoader with batch_size=1 to iterate over samples
        single_sample_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=False)
        sample_diffs = train_on_samples(model, single_sample_loader, loss_fn, LR, device)

        # Step 2: Select top best samples based on improvement ratio
        best_sample_indices = choose_best_samples(sample_diffs, epoch=total_epoch, threshold=THRESHOLD_RATIO)
        print(f"Selected {len(best_sample_indices)} best samples out of {len(full_train_dataset)}")

        # Step 3: Build a new training set from the best samples
        best_texts = [full_train_dataset.texts[i] for i in best_sample_indices]
        best_labels = [full_train_dataset.labels[i] for i in best_sample_indices]
        best_train_dataset = IMDBDataset(best_texts, best_labels, tokenizer, max_length=MAX_LENGTH)
        best_train_loader = DataLoader(best_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # For a given curriculum epoch, training iterations may vary.
        # For simplicity, if curriculum epoch < some value, train longer; then shorter.
        if total_epoch < 1:
            num_learn_iter = 3
        else:
            num_learn_iter = 1

        for epoch in range(num_learn_iter):
            epoch_id = total_epoch * num_learn_iter + epoch + 1
            print("-"*30)
            print(f"Training Epoch: {epoch_id}")
            print("-"*30)
            
            # Train the model
            train_loss, train_acc = train_epoch(epoch_id, model, best_train_loader, loss_fn, optimizer, device)
            # Evaluate on training subset and validation set
            eval_loss_train, eval_acc_train = eval_model(model, best_train_loader, loss_fn, device)
            eval_loss_val, eval_acc_val = eval_model(model, val_loader, loss_fn, device)

            # Logging metrics
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch_id} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Epoch {epoch_id} | Eval (Train Subset): Loss: {eval_loss_train:.4f}, Acc: {eval_acc_train:.4f}")
            print(f"Epoch {epoch_id} | Val: Loss: {eval_loss_val:.4f}, Acc: {eval_acc_val:.4f}")

            # Save best model weights if performance improved
            new_best = False
            if eval_acc_val > best_eval_acc:
                best_eval_acc = eval_acc_val
                best_eval_acc_epoch = epoch_id
                torch.save(model.state_dict(), best_acc_file)
                new_best = True
            if eval_loss_val < best_eval_loss:
                best_eval_loss = eval_loss_val
                best_eval_loss_epoch = epoch_id
                torch.save(model.state_dict(), best_loss_file)
                new_best = True
            if new_best:
                with open(best_text_file, 'w') as f:
                    f.write(f"Best eval acc epoch: {best_eval_acc_epoch}\n")
                    f.write(f"Best eval acc: {best_eval_acc:.4f}\n")
                    f.write(f"Best eval loss epoch: {best_eval_loss_epoch}\n")
                    f.write(f"Best eval loss: {best_eval_loss:.4f}\n")

            # Save weights periodically
            if epoch_id % WEIGHT_SAVE_MODULUS == 0:
                weight_path = os.path.join(weights_folder, f'epoch_{str(epoch_id).zfill(3)}.pth')
                torch.save(model.state_dict(), weight_path)

            # Append results to CSV file
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch_id, current_lr, train_loss, train_acc, eval_loss_val, eval_acc_val])

    # After training: Evaluate best models on validation set (or test set if available).
    print("Evaluating best loss model on validation set:")
    model.load_state_dict(torch.load(best_loss_file))
    best_loss_val, best_loss_acc = eval_model(model, val_loader, loss_fn, device)
    print(f"Best loss model -> Loss: {best_loss_val:.4f}, Acc: {best_loss_acc:.4f}")

    if best_eval_acc_epoch != best_eval_loss_epoch:
        print("Evaluating best accuracy model on validation set:")
        model.load_state_dict(torch.load(best_acc_file))
        best_acc_val, best_acc_acc = eval_model(model, val_loader, loss_fn, device)
        print(f"Best accuracy model -> Loss: {best_acc_val:.4f}, Acc: {best_acc_acc:.4f}")

if __name__ == "__main__":
    main()
