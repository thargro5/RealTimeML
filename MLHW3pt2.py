import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Shakespeare text from your local file
file_path = "/Users/moon-base/Library/Mobile Documents/com~apple~TextEdit/Documents/tinyshakespeare.txt"  # Replace with your local file path
with open(file_path, 'r') as file:
    text = file.read()
    print("Dataset loaded successfully!")

# Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.chars = sorted(list(set(text)))  # Unique characters
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.data = [self.char_to_idx[ch] for ch in text]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return input_seq, target_seq

# Set sequence length
seq_length = 20  # Change to 30 for the other experiment

# Create dataset and dataloader
dataset = ShakespeareDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, model_type="LSTM"):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.embedding = nn.Embedding(input_size, hidden_size)

        if model_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid model type. Choose LSTM or GRU.")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        if self.model_type == "LSTM":
            output, (hidden, cell) = self.rnn(x, hidden)
        else:
            output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model_type == "LSTM":
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

def compute_accuracy(outputs, targets):
    predictions = torch.argmax(outputs, dim=-1)
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    return correct / total

def train_model(model_type, dataloader, seq_length, num_epochs=10, batch_size=64, hidden_size=256, num_layers=2, lr=0.002):
    vocab_size = len(dataloader.dataset.chars)
    model = CharRNN(vocab_size, hidden_size, vocab_size, num_layers, model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += compute_accuracy(outputs, targets).item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        print(f"{model_type} - Seq {seq_length} - Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    execution_time = time.time() - start_time
    model_size = sum(p.numel() for p in model.parameters())
    return avg_loss, avg_acc, execution_time, model_size

results = {}
for model_type in ["LSTM", "GRU"]:
    for seq_length in [20, 30]:
        loss, acc, time_taken, size = train_model(model_type, dataloader, seq_length)
        results[(model_type, seq_length)] = {"Loss": loss, "Accuracy": acc, "Time": time_taken, "Size": size}

print(results)