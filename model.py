import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as functional


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

    
class Sequence:
    def __init__(self, sentence, inputs, states, score, max_len):
        self.sentence = sentence
        self.inputs = inputs
        self.states = states
        self.score = score
        self._max_len = max_len

    def is_done(self):
        size = len(self)
        return size == self._max_len or (size > 1 and self.sentence[-1] == 1)

    def __len__(self):
        return len(self.sentence)

    def __repr__(self):
        return "Sequence(sentence={}, score={})".format(self.sentence, self.score)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, beam_size=0):
        super(DecoderRNN, self).__init__()
        self.beam_size = beam_size
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
        if self.beam_size == 0:
            return self.sample_greedely(inputs, states, max_len)
        else:
            return self.sample_beam_search(inputs, states, max_len)
        
    def sample_beam_search(self, inputs, states, max_len):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sequences = [Sequence(list(), inputs, states, 1.0, max_len)]
        while not all(sequence.is_done() for sequence in sequences):
            all_candidates = list()
            for sequence in sequences:
                if sequence.is_done():
                    all_candidates.append(sequence)
                else:
                    hidden, states = self.rnn(sequence.inputs, sequence.states)
                    predictions = functional.softmax(self.fc(hidden.squeeze(1)), dim=1)[0]
                    for token, p in enumerate(predictions):
                        embedded_token = self.embedding(torch.Tensor([token]).long().to(predictions.device)).unsqueeze(1)
                        sentence = sequence.sentence + [token]
                        # subtract from one to avoid vanishing score as the prediction values of the starting toking is 
                        # almost always near one this results in very low scores. 
                        score = sequence.score * (1 - math.log(p.item()))    
                        candidate = Sequence(sentence, embedded_token, states, score, max_len)
                        all_candidates.append(candidate)

            sequences = sorted(all_candidates, key=lambda c: c.score)[:self.beam_size]
            
        return sequences[0].sentence
    
    def sample_greedely(self, inputs, states, max_len):
        sentence = list()
        for _ in range(max_len):
            hidden, states = self.rnn(inputs, states)
            predictions = functional.softmax(self.fc(hidden.squeeze(1)), dim=1)
            _, best_pred = predictions.max(1)
            inputs = self.embedding(best_pred).unsqueeze(1) # prepare next input recursion
            token = best_pred.item()
            sentence.append(token)
            if token == 1: # Encountered <end> token
                break;
        return sentence
        