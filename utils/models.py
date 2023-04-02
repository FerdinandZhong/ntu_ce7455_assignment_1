import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        pretrained_weight=None,
        pretraining_freeze=True,
        dropout_rate = 0.25
    ):
        super().__init__()

        if pretrained_weight is None:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weight, freeze=pretraining_freeze
            )

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]

        _, hidden = self.rnn(embedded)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        if self.dropout_rate > 0:
            hidden = self.dropout(hidden)

        return self.fc(hidden.squeeze(0))


class MLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim_list, output_dim) -> None:
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.fc_layers = []
        last_hidden_dim = embedding_dim
        for hidden_dim in hidden_dim_list:
            self.fc_layers.append(nn.Linear(last_hidden_dim, hidden_dim))
            last_hidden_dim = hidden_dim
        
        self.fc_layers = nn.ModuleList(self.fc_layers)

        self.final_fc = nn.Linear(last_hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)

        x = torch.mean(embedded, 0)

        for layer in self.fc_layers:
            x = layer(x)

        return self.final_fc(x)


class CNN(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        filter_sizes=[1,2,3],
        dropout_rate = 0.25,
        pretrained_weight=None,
        pretraining_freeze=True,
    ) -> None:
        super().__init__()

        if pretrained_weight is None:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weight, freeze=pretraining_freeze
            )

        self.cov_1d = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=fs
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim * len(filter_sizes), output_dim)

    def forward(self, text):
        embedded = self.embedding(text).permute(1, 2, 0)
        # embedded = [batch_size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.cov_1d]
        # conved = [batch size, hidden_dim, sent len - filter_sizes[n] + 1]

        # let every sentence in every filter become one number
        pooled_1 = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, hidden_dim]

        cat = self.dropout(torch.cat(pooled_1, dim=1))
        # cat = [batch size, hidden_dim * len(filter_sizes)]

        return self.fc(cat)


class LSTM(nn.Module):
    def __init__(self, 
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        dropout_rate = 0.25,
        is_bidirectional=False,
        pretrained_weight=None,
        pretraining_freeze=True,
    ) -> None:
        super().__init__()
        self.is_bidirectional = is_bidirectional

        if pretrained_weight is None:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weight, freeze=pretraining_freeze
            )

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=is_bidirectional)
        self.dropout = nn.Dropout(dropout_rate)

        if not is_bidirectional:
            self.fc = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]

        _, hidden = self.lstm(embedded)

        if isinstance(hidden, Tuple):
            hidden = hidden[0]

        # hidden = [1, batch size, hid dim]

        if self.is_bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :] # equals hidden.squeeze(0)
        
        hidden = self.dropout(hidden)

        return self.fc(hidden)


