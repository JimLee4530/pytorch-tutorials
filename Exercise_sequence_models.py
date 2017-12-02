import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

# lstm = nn.LSTM(3, 3)
#
# inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)]
#
# hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(1, 1, 3)))
#
# for i in inputs:
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
#
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.rand(1, 1, 3)))
# out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)

def prepare_sequence(seq, to_ix):
    word_idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(word_idxs)
    return autograd.Variable(tensor)

def prepare_word(word, to_ix):
    char_idxs = [to_ix[c] for c in word]
    tensor = torch.LongTensor(char_idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".lower().split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

char_to_ix ={}

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)

print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

CHAR_HIDDEN_DIM = 3
CHAR_EMBEDDING_DIM = 6
WORD_HIDDEN_DIM = 6
WORD_EMBEDDING_DIM = 5

class LSTMTagger(nn.Module):
    def __init__(self, word_embedding, char_embedding, word_hidden_dim, char_hidden_dim, vocab_size, tagset_size, char_size):
        super(LSTMTagger, self).__init__()
        #CHAR LSTM
        self.char_hidden_dim = char_hidden_dim
        self.char_embeddings = nn.Embedding(char_size, char_embedding)
        self.char_lstm = nn.LSTM(char_embedding, char_hidden_dim)
        self.char_hidden = self.init_char_hidden()

        # WORD LSTM
        self.word_hidden_dim = word_hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding)
        self.word_lstm  = nn.LSTM(word_embedding+char_hidden_dim, word_hidden_dim)
        self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
        self.word_hidden = self.init_word_hidden()

    def init_word_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.word_hidden_dim)), autograd.Variable(torch.zeros(1, 1, self.word_hidden_dim)))

    def init_char_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.char_hidden_dim)), autograd.Variable(torch.zeros(1, 1, self.char_hidden_dim)))

    def forward(self, sentence_in, sentence):
        char_final_hidden = []
        for word in sentence:
            chars = prepare_word(word, char_to_ix)
            char_embeds = self.char_embeddings(chars)
            _, self.char_hidden = self.char_lstm(char_embeds.view(len(chars), 1, -1), self.char_hidden)
            char_final_hidden.append(self.char_hidden[0].view(-1))
            self.char_hidden = self.init_char_hidden()

        embeds = self.word_embeddings(sentence_in)
        char_final_hidden =  torch.stack(char_final_hidden)

        embeds = torch.cat((embeds, char_final_hidden), 1)
        # print(embeds.size())
        lstm_out, self.hidden = self.word_lstm(embeds.view(len(sentence_in), 1, -1), self.word_hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence_in), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs, training_data[0][0])
print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        # init
        model.zero_grad()
        model.char_hidden = model.init_char_hidden()
        model.word_hidden = model.init_word_hidden()

        # prepare sequence
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # run forward
        tag_scores = model(sentence_in, sentence)

        # compute the loss, grad
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs, training_data[0][0])
print(tag_scores)

