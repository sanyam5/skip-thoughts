import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from config import *
import random
import numpy as np


class Packer:

    def __init__(self):
        pass

    @staticmethod
    def do_order(original, order):

        ordered = [None] * len(original)

        for x, y in enumerate(order):
            ordered[x] = original[y]

        return ordered

    @staticmethod
    def back_order(ordered, order):

        original = [None] * len(ordered)

        for x, y in enumerate(order):
            original[y] = ordered[x]

        return original

    @staticmethod
    def _pad(ordered):
        maxwords = MAXLEN + 1
        lengths = [len(sent) for sent in ordered]

        zero_pad = Variable(torch.zeros((maxwords))).long()
        if USE_CUDA is not None:
            gpu_device = ordered[0].get_device()
            zero_pad = zero_pad.cuda(gpu_device)
            

        padded = []
        for sent in ordered:
            # sent = (num_words)
            num_words = len(sent)
            padded.append(torch.cat([sent, zero_pad])[:-num_words])

        padded = torch.stack(padded)  # (G*B, maxwords)
        padded = padded[:, :MAXLEN]   # (G*B, maxlen)

        return padded, lengths

    @classmethod
    def pad(cls, sentence_groups):
        """

        Args:
            sentence_groups: is a list of list of Variables, G*[B*[ (varying between 1 to MAXLEN) ]]

        Returns:

        """

        flat = []
        for sentences in sentence_groups:
            flat.extend(sentences)

        # indices ordered in decreasing order of the lengths of data they point to.
        indices = list(range(len(flat)))
        order = sorted(indices, key=lambda idx: len(flat[idx]), reverse=True)
        ordered = cls.do_order(flat, order)
        padded, lengths = cls._pad(ordered)  # padded = (G*B, maxwords)

        return padded, order, lengths


class Encoder(nn.Module):
    thought_size = 1200
    word_size = 620

    def __init__(self):
        super().__init__()
        self.word2embd = nn.Embedding(VOCAB_SIZE, self.word_size)
        self.lstm = nn.LSTM(self.word_size, self.thought_size)

    def forward(self, sentences, lengths):
        # sentences = (batch_size, maxlen)

        word_embeddings = F.tanh(self.word2embd(sentences))  # (batch, maxlen, word_size)
        packed = torch.nn.utils.rnn.pack_padded_sequence(word_embeddings, lengths=lengths, batch_first=True)
        _, (thoughts, _) = self.lstm(packed)
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
        # word_embeddings = # (batch, maxlen, word_size)

        assert word_embeddings.size()[1] == MAXLEN
        assert thoughts.size()[0] == word_embeddings.size()[0]

        thoughts = thoughts.repeat(MAXLEN, 1, 1)  # (maxlen, batch, thought_size)
        word_embeddings = word_embeddings.transpose(0, 1)  # (maxlen, batch, word_size)

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
        lengths = lengths.data.numpy()
        for i, l in enumerate(lengths):
            for j in range(l):
                mask[i, j] = 1
        return Variable(mask).cuda(var.get_device())

    def forward(self, sentence_groups):
        # sentences = G * [B * [1 to maxlen]]]
        group_size = len(sentence_groups)
        batch_size = len(sentence_groups[0])

        padded, order, deranged_lengths = Packer.pad(sentence_groups)  # (G*B, maxlen)
        thoughts, word_embeddings = self.encoder(padded, deranged_lengths)  # thoughts = (G*B, thought_size), word_embeddings = (G*B, maxlen, word_size)

        thoughts = torch.stack(Packer.back_order(thoughts, order)).view(group_size, batch_size, Encoder.thought_size)  # (G, B, thought_size)
        word_embeddings = torch.stack(Packer.back_order(word_embeddings, order)).view(group_size, batch_size, MAXLEN, Encoder.word_size)  # (G, B, maxlen, word_size)
        
        deranged_lengths = Variable(torch.from_numpy(np.array(deranged_lengths))).long()
        lengths = torch.stack(Packer.back_order(deranged_lengths, order)).view(group_size, batch_size)  # (G, B)

        for i, batch in enumerate(sentence_groups):
            for j, sent in enumerate(batch):
                assert lengths[i, j].data[0] == len(sent)

        assert group_size == 1
        thoughts, word_embeddings, lengths = thoughts[0], word_embeddings[0], lengths[0]
        prev_pred, next_pred = self.decoders(thoughts, word_embeddings)  # both = (batch-1, maxlen, VOCAB_SIZE)
        
        padded = padded.contiguous()
        
        prev_mask = self.create_mask(prev_pred, lengths[:-1])
        next_mask = self.create_mask(next_pred, lengths[1:])
        
        masked_prev_pred = prev_pred * prev_mask
        masked_next_pred = next_pred * next_mask
        
        prev_loss = F.cross_entropy(masked_prev_pred.view(-1, VOCAB_SIZE), padded[:-1, :].view(-1))
        next_loss = F.cross_entropy(masked_next_pred.view(-1, VOCAB_SIZE), padded[1:, :].view(-1))

        loss = prev_loss + next_loss
        
        _, prev_pred_ids = prev_pred[0].max(1)
        _, next_pred_ids = next_pred[0].max(1)
        

        return loss, padded[0], padded[1], prev_pred_ids, next_pred_ids













