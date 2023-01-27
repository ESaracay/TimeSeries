import torch
from torch import nn

import numpy as np

samples = ['hey how are you','good i am fine','have a nice day']
chars = set(''.join(samples))
int2char = dict(enumerate(chars))
char2int = {char: ind for ind,char in int2char.items()}

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

def clean_data(samples):
    input_seq = []
    target_seq = []
    maxlen = len(max(samples, key=len))
    # Same padding to run in batches
    for i in range(len(samples)):
        while(len(samples[i]) < maxlen):
            samples[i] += ' '

    # Everything but the last output
    for i in range(len(samples)):
        input_seq.append(samples[i][:-1])
        target_seq.append(samples[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

    # Convert characters into integers
    for i in range(len(samples)):
        input_seq[i] = [char2int[char] for char in input_seq[i]]
        target_seq[i] = [char2int[char] for char in target_seq[i]]

    # One hot encode every character
    input_seq = one_hot_encode(input_seq, dict_size=len(char2int), seq_len=maxlen - 1, batch_size=len(samples))

    # Convert to torch tensors
    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)

    return (input_seq, target_seq)

# Input shape --> (Batch Size, Sequence Length, One Hot Encoding Size)
def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

def setup_gpu():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()
        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

def main():
    device = setup_gpu()
    dict_size = len(char2int)
    input_seq, target_seq = clean_data(samples)
    model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)

    # Hyperparameters
    n_epochs = 100
    lr = 0.01

    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        input_seq.to(device)
        output, hidden = model(input_seq)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
    
        if epoch%10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

if __name__ == "__main__":
    main()