import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
import optuna
import flask
from flask import request, jsonify

class Config:
    def __init__(self):
        self.vocab_size = 30000
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.dropout = 0.1
        self.max_len = 500
        self.lr = 0.0001
        self.batch_size = 64
        self.num_epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return x.view(x.size(0), -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        query = self.split_heads(self.query_layer(query))
        key = self.split_heads(self.key_layer(key))
        value = self.split_heads(self.value_layer(value))

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.depth)
        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(query.size(0), -1, self.d_model)
        return self.fc_out(context)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout(ffn_output))

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_len)
        self.encoder_layers = nn.ModuleList([TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout) for _ in range(config.num_layers)])
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.fc_out(x)

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, model.config.vocab_size), inputs.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.config.vocab_size), inputs.view(-1))
            running_loss += loss.item()
    return running_loss / len(val_loader)

def objective(trial):
    config = Config()
    model = TransformerModel(config).to(config.device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data = ["Some example text", "Another example text"]  # Example data
    train_dataset = TextDataset(train_data, tokenizer, config.max_len)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        train_loss = train(model, criterion, optimizer, train_loader, config.device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")

    return train_loss

def create_app():
    app = flask.Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        text = data['text']
        model = TransformerModel(Config()).to(Config().device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=500)
        output = model(inputs.input_ids.to(Config().device))
        predicted = output.argmax(dim=-1)
        return jsonify({'prediction': predicted.tolist()})

    return app

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    app = create_app()
    app.run(debug=True)
