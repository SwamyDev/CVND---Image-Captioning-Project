import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]  # don't pass in the <end> token - the model determines the <end>
        embedded = torch.cat((features.unsqueeze(1), self.embedding(captions)), 1)
        hidden, _ = self.rnn(embedded)
        outputs = self.fc(hidden)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = list()
        for _ in range(max_len):
            hidden, states = self.rnn(inputs, states)
            predictions = self.fc(hidden.squeeze(1))
            _, best_pred = predictions.max(1)
            inputs = self.embedding(best_pred).unsqueeze(1) # prepare next input recursion
            token = best_pred.item()
            sentence.append(token)
            if token == 1: # Encountered <end> token
                break;
            
        return sentence;