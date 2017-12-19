import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from config import *
import random
import numpy as np


class Encoder(nn.Module):
    thought_size = 1200
    word_size = 620

    @staticmethod
    def reverse_variable(var):
        idx = [i for i in range(var.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))

        if USE_CUDA:
            idx = idx.cuda(CUDA_DEVICE)

        inverted_var = var.index_select(0, idx)
        return inverted_var

    def __init__(self):
        super().__init__()
        self.word2embd = nn.Embedding(VOCAB_SIZE, self.word_size)
        self.lstm = nn.LSTM(self.word_size, self.thought_size)

    def forward(self, sentences):
        # sentences = (batch_size, maxlen)

        sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)

        word_embeddings = F.tanh(self.word2embd(sentences))  # (maxlen, batch_size, word_size)

        # read embeddings in reverse, required to move paddinng before.
        rev = self.reverse_variable(word_embeddings)

        _, (thoughts, _) = self.lstm(rev)
        thoughts = thoughts[-1]  # (batch, thought_size)

        return thoughts, word_embeddings


class DuoDecoder(nn.Module):

    word_size = Encoder.word_size

    def __init__(self):
        super().__init__()
        self.prev_lstm = nn.LSTM(Encoder.thought_size + self.word_size, self.word_size)
        self.next_lstm = nn.LSTM(Encoder.thought_size + self.word_size, self.word_size)
        self.worder = nn.Linear(self.word_size, VOCAB_SIZE)

    def forward(self, thoughts, word_embeddings):
        # thoughts = (batch_size, Encoder.thought_size)
        # word_embeddings = # (maxlen, batch, word_size)

        batch_size = thoughts.size()[0]
        
        assert word_embeddings.size()[0] == MAXLEN
        assert word_embeddings.size()[1] == batch_size

        thoughts = thoughts.repeat(MAXLEN, 1, 1)  # (maxlen, batch, thought_size)
        a, b, c = thoughts.size()
        assert a == MAXLEN and b == batch_size and c == Encoder.thought_size

        prev_thoughts = thoughts[:, :-1, :]  # (maxlen, batch-1, thought_size)
        next_thoughts = thoughts[:, 1:, :]   # (maxlen, batch-1, thought_size)

        # teacher forcing.
        prev_word_embeddings = word_embeddings[:, :-1, :]  # (maxlen, batch-1, word_size)
        next_word_embeddings = word_embeddings[:, 1:, :]  # (maxlen, batch-1, word_size)
        # delay the embeddings by one timestep
        delayed_prev_word_embeddings = torch.cat([0 * prev_word_embeddings[-1:, :, :], prev_word_embeddings[:-1, :, :]])
        delayed_next_word_embeddings = torch.cat([0 * next_word_embeddings[-1:, :, :], next_word_embeddings[:-1, :, :]])

        prev_pred_embds, _ = self.prev_lstm(torch.cat([next_thoughts, delayed_prev_word_embeddings], dim=2))  # (maxlen, batch-1, embd_size)
        next_pred_embds, _ = self.prev_lstm(torch.cat([prev_thoughts, delayed_next_word_embeddings], dim=2))  # (maxlen, batch-1, embd_size)

        # predict actual words
        a, b, c = prev_pred_embds.size()
        prev_pred = self.worder(prev_pred_embds.view(a*b, c)).view(a, b, -1)  # (maxlen, batch-1, VOCAB_SIZE)
        a, b, c = next_pred_embds.size()
        next_pred = self.worder(next_pred_embds.view(a*b, c)).view(a, b, -1)  # (maxlen, batch-1, VOCAB_SIZE)

        prev_pred = prev_pred.transpose(0, 1).contiguous()  # (batch-1, maxlen, VOCAB_SIZE)
        next_pred = next_pred.transpose(0, 1).contiguous()  # (batch-1, maxlen, VOCAB_SIZE)

        return prev_pred, next_pred


class UniSkip(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoders = DuoDecoder()

    def create_mask(self, var, lengths):
        mask = var.data.new().resize_as_(var.data).fill_(0)
#         print("lengths", lengths)
        for i, l in enumerate(lengths):
            for j in range(l):
                mask[i, j] = 1
        return Variable(mask).cuda(var.get_device())

    def forward(self, sentence_groups, lengths):
        # sentences = (G, B, maxlen)
        # lengths = (G, B)
        group_size, batch_size, _ = sentence_groups.size()

        assert group_size == 1
        lengths = lengths[0]
        sentences = sentence_groups.view(group_size * batch_size, -1)

        thoughts, word_embeddings = self.encoder(sentences)  # thoughts = (G*B, thought_size), word_embeddings = (G*B, maxlen, word_size)

        prev_pred, next_pred = self.decoders(thoughts, word_embeddings)  # both = (batch-1, maxlen, VOCAB_SIZE)
        
        prev_mask = self.create_mask(prev_pred, lengths[:-1])
        next_mask = self.create_mask(next_pred, lengths[1:])
        
        masked_prev_pred = prev_pred * prev_mask
        masked_next_pred = next_pred * next_mask
        
        prev_loss = F.cross_entropy(masked_prev_pred.view(-1, VOCAB_SIZE), sentences[:-1, :].view(-1))
        next_loss = F.cross_entropy(masked_next_pred.view(-1, VOCAB_SIZE), sentences[1:, :].view(-1))

        loss = prev_loss + next_loss
        
        _, prev_pred_ids = prev_pred[0].max(1)
        _, next_pred_ids = next_pred[0].max(1)

        return loss, sentences[0], sentences[1], prev_pred_ids, next_pred_ids













