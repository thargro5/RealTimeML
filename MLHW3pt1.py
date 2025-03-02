import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Given text sequence
text = """Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."""

# Create character mappings
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert text to numerical representation
encoded_text = np.array([char_to_idx[ch] for ch in text])

class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx : idx + self.seq_length], dtype=torch.long),
            torch.tensor(self.data[idx + 1 : idx + self.seq_length + 1], dtype=torch.long),
        )

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type="RNN"):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.embedding = nn.Embedding(input_size, hidden_size)

        if model_type == "RNN":
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Invalid model type. Choose from RNN, LSTM, or GRU.")

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
            return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                    torch.zeros(1, batch_size, self.hidden_size, device=device))
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

def compute_accuracy(outputs, targets):
    predictions = torch.argmax(outputs, dim=-1)
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    return correct / total

def train_model(model_type, seq_length, num_epochs=10, batch_size=32, hidden_size=128, lr=0.002):
    dataset = TextDataset(encoded_text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CharRNN(len(chars), hidden_size, len(chars), model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            batch_size = inputs.size(0)  # Dynamically adjust batch size
            hidden = model.init_hidden(batch_size)
            
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, len(chars)), targets.view(-1))
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
for model_type in ["RNN", "LSTM", "GRU"]:
    for seq_length in [10, 20, 30]:
        loss, acc, time_taken, size = train_model(model_type, seq_length)
        results[(model_type, seq_length)] = {"Loss": loss, "Accuracy": acc, "Time": time_taken, "Size": size}

print(results)