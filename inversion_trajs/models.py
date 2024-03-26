import torch
import torch.nn as nn
    
class TransformerLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_size, num_heads, dropout=dropout)    # self-attention layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, input):
        """
        Input:
        input (seq_len, batch, input_size): input tensor
        ---
        Output:
        output (seq_len, batch, input_size): output tensor
        """
        # Self-attention layer
        # get the query, key, value
        query = self.query(input)   # (seq_len, batch, input_size)
        key = self.key(input)   # (seq_len, batch, input_size)
        value = self.value(input)   # (seq_len, batch, input_size)
        output, _ = self.self_attention(query, key, value)  # (seq_len, batch, input_size)

        # Feedforward layer
        output = self.norm(output + input)  # Add and norm    output: (batch, seq_len, input_size)
        output = nn.functional.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.norm(output + input)  # Add and norm

        # output = output.permute(1, 0, 2)  # Change back to (seq_len, batch, input_size)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_heads = 1):
        super(TransformerDecoder, self).__init__()

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(input_size, hidden_size, num_heads, dropout) for _ in range(1)
        ])
        self.linear = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = 1

    def forward(self, input):
        """
        Input:
        input (batch, embedding_size): input embeddings, 
            but because I only have one embedding for the whole sequence, so the input is (seq_len=1, batch, embedding_size)
        ---
        Output:
        output (seq_len, batch, embedding_size): output tensor
        """
        input = input.unsqueeze(0)  # Add seq_len dimension, now input is (seq_len=1, batch, embedding_size)
        # now the seq_len is 1, but I want to change the seq_len to 20, so I will repeat the input 20 times
        input = input.repeat(20, 1, 1)  # (seq_len=20, batch, embedding_size)

        for transformer_layer in self.transformer_layers:
            input = transformer_layer(input)
        input = self.dropout(input)
        output = self.linear(input)   # (seq_len, batch, hidden_size)
        h = None
        return output, h
    

class StackingGRUCell(nn.Module):
    """
    Multi-layer GRU Cell
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()

        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.linear = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, input, h = [None, None, None]):
        """
        Input:
        input (batch, embedding_size): input embeddings
        h (num_layers, batch, hidden_size): input hidden state
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (batch, embedding_size)"
        output = []
        for i in range(20):
            o, h = self.rnn(input, h)
            o = self.dropout(o)
            input = self.linear(o)
            output.append(o)
        output = torch.stack(output)
        return output, h
    

'''********* try LSTM ***********'''    
class LSTMDeocder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMDeocder, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, input):
        input = input.unsqueeze(0)
        input = input.repeat(20, 1, 1)

        output, h = self.lstm(input)
        output = self.linear(output)
        output = self.dropout(output)
        return output, h
