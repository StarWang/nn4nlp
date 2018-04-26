import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class StackedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0, padding=False):
        super(StackedBiLSTM, self).__init__()
        self.padding = padding
        self.dropout_prob = dropout_prob
        self.dropouts = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                lstm_input_size = input_size
            else:
                lstm_input_size = 2 * hidden_size
            self.layers.append(nn.LSTM(lstm_input_size, hidden_size, 1, bidirectional=True, batch_first=True));
            self.dropouts.append(nn.Dropout(dropout_prob))

    def forward(self, input_, mask):
        # input_: B x len x dim
        # output: B x len x 2*dim
        if not self.training:
            return self._forward_padded(input_, mask)

        return self._forward_unpadded(input_, mask)

    # return self._forward_padded(input_, mask)

    def _forward_unpadded(self, input_, mask):
        # input_: B x len x dim
        # output: B x len x 2*dim
        lstm_output = input_
        for i in range(len(self.layers)):
            if self.dropout_prob > 0:
                lstm_output = self.dropouts[i](lstm_output);
            lstm_output = self.layers[i](lstm_output)[0]

        output = lstm_output
        return output.contiguous()

    def _forward_padded(self, input_, mask):
        # input_: B x len x dim
        # output: B x len x 2*dim
        lengths = mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        input_ = input_.index_select(0, idx_sort)
        input_ = nn.utils.rnn.pack_padded_sequence(input_, lengths, batch_first=True)

        # Encode all layers
        for i in range(len(self.layers)):
            lstm_input = lstm_output if i > 0 else input_
            if self.dropout_prob > 0:
                lstm_input_1 = self.dropouts[i](lstm_input.data)
                lstm_input = nn.utils.rnn.PackedSequence(lstm_input_1, lstm_input.batch_sizes)
            # lstm_input = nn.utils.rnn.PackedSequence(lstm_input_1, lstm_input.batch_sizes, batch_first = True)
            # lstm_input = F.dropout(lstm_input.data,
            #                           p=self.dropout_prob,
            #                           training=self.training)
            # lstm_input = nn.utils.rnn.PackedSequence(lstm_input,
            #                                         lstm_input.batch_sizes)
            lstm_output = self.layers[i](lstm_input)[0]

        output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        output = output.index_select(0, idx_unsort)

        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        return output.contiguous()


class SequenceAttentionMM(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0, name=''):
        super(SequenceAttentionMM, self).__init__()
        self.W1 = nn.Linear(input_size, output_size)
        self.activation = F.relu
        self.name = name  # use the name field to debug
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p=dropout_prob)

    # u: B x len1 x Dim
    # v: B x len2 x Dim
    # output: B x len1 x Dim
    def forward(self, u, v, v_mask):
        # compute alpha_i
        u_ = self.activation(self.W1(u))  # u_: B x len1 x output_size
        v_ = self.activation(self.W1(v))  # v_: B x len2 x output_size
        # print(self.name)
        # print("u size:", u.size())
        # print("v size:", v.size())
        # print("u_ size:", u_.size())
        # print("v_ size:", v_.size())

        alpha = u_.bmm(v_.permute(0, 2, 1))  # alpha: B x len1 x len2
        v_mask = v_mask.unsqueeze(1)  # v_mask: B x 1 x len2
        # print("v_mask size:", v_mask.size())
        # print("alpha size:", alpha.size())
        # print()
        alpha.data.masked_fill_(v_mask.expand(alpha.size()).data, -float('inf'))
        alpha = F.softmax(alpha, dim=2)

        output = alpha.bmm(v)
        if self.dropout_prob > 0:
            output = self.dropout(output)
        return output


class SequenceAttentionMV(nn.Module):
    def __init__(self, input_size, output_size, name=''):
        super(SequenceAttentionMV, self).__init__()
        self.W1 = nn.Linear(input_size, output_size)  # input_size Dim2, output_size Dim1
        self.name = name  # use the name field to debug

    # u: B x len1 x Dim1
    # v: B x Dim2
    # output: B x Dim1
    def forward(self, u, v, u_mask):
        # compute alpha_i
        v_ = self.W1(v)  # v_: B x Dim1

        alpha = u.bmm(v_.unsqueeze(2)).permute(0, 2, 1)  # alpha: B x 1 x len1
        u_mask = u_mask.unsqueeze(1)
        alpha.data.masked_fill_(u_mask.data, -float('inf'))
        alpha = F.softmax(alpha, dim=2)

        output = alpha.bmm(u).squeeze(1)
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_size, name=''):
        super(SelfAttention, self).__init__()
        self.W2 = nn.Linear(input_size, 1)
        self.name = name

    # u: B x len x Dim
    # output: B x Dim
    def forward(self, u, u_mask):
        u_ = self.W2(u).squeeze(2)
        # print(u_mask.size(), u_.size())
        u_.data.masked_fill_(u_mask.data, -float('inf'))
        alpha = F.softmax(u_.unsqueeze(1), dim=2)  # alpha: B x 1 x len

        return alpha.bmm(u).squeeze(1)


class AllEmbedding(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, ner_vocab_size, rel_vocab_size, char_vocab_size,
                 vocab_embedding_dim=300, pos_embedding_dim=12, ner_embedding_dim=8, rel_embedding_dim=10,
                 char_embedding_dim=200, dropout_prob=0):
        super(AllEmbedding, self).__init__()
        self.wordEmbedding = nn.Embedding(word_vocab_size, vocab_embedding_dim, padding_idx=0)
        self.posEmbedding = nn.Embedding(pos_vocab_size, pos_embedding_dim, padding_idx=0)
        self.nerEmbedding = nn.Embedding(ner_vocab_size, ner_embedding_dim, padding_idx=0)
        self.relEmbedding = nn.Embedding(rel_vocab_size, rel_embedding_dim, padding_idx=0)
        self.charEmbedding = CharEmbed(char_vocab_size, embed_dim=char_embedding_dim)
        self.wordEmbedding.weight.data.fill_(0)
        self.wordEmbedding.weight.data[:2].normal_(0, 0.1)  # only initialze the first two indices
        self.posEmbedding.weight.data.normal_(0, 0.1)
        self.nerEmbedding.weight.data.normal_(0, 0.1)
        self.relEmbedding.weight.data.normal_(0, 0.1)
        self.dropout_prob = dropout_prob
        self.dropouts = nn.ModuleList()
        for i in range(9):
            self.dropouts.append(nn.Dropout(self.dropout_prob))

    def loadGloveEmbedding(self, wordIdxWeight):
        for wordIdx, weight in wordIdxWeight.items():
            self.wordEmbedding.weight.data[wordIdx].copy_(pretrained)

    def forward(self, indices):
        p, q, c, pPos, pNer, qPos, pQRel, pCRel, pChars, qChars, cChars = indices
        pEmb, qEmb, cEmb = self.wordEmbedding(p), self.wordEmbedding(q), self.wordEmbedding(c)
        pPosEmb, pNerEmb, qPosEmb = self.posEmbedding(pPos), self.nerEmbedding(pNer), self.posEmbedding(qPos)
        pQRelEmb, pCRelEmb = self.relEmbedding(pQRel), self.relEmbedding(pCRel)
        pCharsEmb, qCharsEmb, cCharsEmb = self.charEmbedding(pChars), self.charEmbedding(qChars), self.charEmbedding(
            cChars)
        print (pEmb.size(), pCharsEmb.size())
        pEmb = torch.cat([pEmb, pCharsEmb], dim=2)
        qEmb = torch.cat([qEmb, qCharsEmb], dim=2)
        cEmb = torch.cat([cEmb, cCharsEmb], dim=2)

        if self.dropout_prob > 0:
            pEmb, qEmb, cEmb = self.dropouts[0](pEmb), self.dropouts[1](qEmb), self.dropouts[2](cEmb)
            pPosEmb, pNerEmb, qPosEmb = self.dropouts[3](pPosEmb), self.dropouts[4](pNerEmb), self.dropouts[5](qPosEmb)
            pQRelEmb, pCRelEmb = self.dropouts[6](pQRelEmb), self.dropouts[7](pCRelEmb)

        return pEmb, qEmb, cEmb, pPosEmb, pNerEmb, qPosEmb, pQRelEmb, pCRelEmb


class CharEmbed(nn.Module):
    def __init__(self, vocab_size, window_size=5, embed_dim=50, output_dim=200):
        super(CharEmbed, self).__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.char_embed.weight.data.normal_(0, 0.1)
        self.conv = nn.Conv2d(1, output_dim, (window_size, embed_dim), padding=((window_size - 1) // 2, 0))

    # input: batch_size*sent_len*char_num
    def forward(self, char_sequences):
        result = []
        for char_seq in char_sequences:
            print (char_seq.size())
            # batch_size*word_len*embed_dim
            char_embedding = self.char_embed(char_seq)
            # batch_size*c_out*conv_word_len [*conv_embed_dim(1)]
            conv_result = self.conv(char_embedding.unsqueeze(1)).squeeze(3)
            print (conv_result.size())
            # batch_size*c_out
            word_char_embed = F.max_pool1d(conv_result, conv_result.size(2)).squeeze(2)
            print (word_char_embed.size())
            print ()
            # batch_size*1*c_out
            result.append(word_char_embed)
        return torch.cat(result, dim=1)
