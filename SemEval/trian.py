import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import StackedBiLSTM, SequenceAttentionMM, SequenceAttentionMV, SelfAttention, AllEmbedding
from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class TriAN(nn.Module):
    def __init__(self, args):
        super(TriAN, self).__init__()
        self.args = args
        self.embedding_dim = 300
        # 0. Embedding layer
        self.embeddings = AllEmbedding(len(vocab), len(pos_vocab), len(ner_vocab), len(rel_vocab), self.embedding_dim, args.pos_emb_dim, args.ner_emb_dim, args.rel_emb_dim, args.dropout_emb)
        self.p_q_embedder = SequenceAttentionMM(self.embedding_dim, self.embedding_dim)
        self.c_q_embedder = SequenceAttentionMM(self.embedding_dim, self.embedding_dim)
        self.c_p_embedder = SequenceAttentionMM(self.embedding_dim, self.embedding_dim)

        """
        passage = [p_embed, p_q_embed, p_pos_embed, p_ner_embed, p_q_rel_embed, p_c_rel_embed, f_tensor]
        question = [q_embed, q_pos_embed]
        choice = [c_embed, c_q_embed, c_p_embed]
        """
        p_input_size = 2 * self.embedding_dim + args.pos_emb_dim + args.ner_emb_dim + 2 * args.rel_emb_dim + 5
        q_input_size = self.embedding_dim + args.pos_emb_dim
        c_input_size = 3 * self.embedding_dim

        # 1. RNN layers for passage, question, choice
        self.p_rnn = StackedBiLSTM(p_input_size, args.hidden_size, args.doc_layers, dropout_prob = 0, padding = args.rnn_padding)
        self.q_rnn = StackedBiLSTM(q_input_size, args.hidden_size, 1, dropout_prob = 0, padding = args.rnn_padding)
        self.c_rnn = StackedBiLSTM(c_input_size, args.hidden_size, 1, dropout_prob = 0, padding = args.rnn_padding)

        # 2. Attention layers for question, passage-question, choice
        self.q_qAttn = SelfAttention(2 * args.hidden_size)
        self.p_qAttn = SequenceAttentionMV(2 * args.hidden_size, 2 * args.hidden_size)
        self.c_cAttn = SelfAttention(2 * args.hidden_size)
        
        p_attn_out_size = 2 * args.hidden_size
        c_attn_out_size = 2 * args.hidden_size
        q_attn_out_size = 2 * args.hidden_size
        # 3. Merge hiddens into final answer
        self.p_c_interact = nn.Linear(p_attn_out_size, c_attn_out_size)
        self.q_c_interact = nn.Linear(q_attn_out_size, c_attn_out_size)

    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, p_q_relation, p_c_relation):
        p_emb, q_emb, c_emb, p_pos_emb, p_ner_emb, q_pos_emb, p_q_rel_emb, p_c_rel_emb = self.embeddings([p, q, c, p_pos, q_pos, p_q_relation, p_c_relation])
        p_q_emb = self.p_q_embedder(p_emb, q_emb, q_mask)
        c_q_emb = self.c_q_embedder(c_emb, q_emb, q_mask)
        c_p_emb = self.c_p_embedder(c_emb, p_emb, c_mask)

        p_rnn_in = torch.cat([p_emb, p_q_emb, p_pos_emb, p_ner_emb, f_tensor], dim = 2)
        q_rnn_in = torch.cat([q_emb, q_pos_emb], dim = 2)
        c_rnn_in = torch.cat([c_emb, c_q_emb, c_p_emb], dim = 2)

        p_rnn_out = self.p_rnn(p_rnn_in, p_mask)
        q_rnn_out = self.q_rnn(q_rnn_in, q_mask)
        c_rnn_out = self.c_rnn(c_rnn_in, c_mask)

        q_attn_out = self.q_qAttn(q_rnn_out)
        p_attn_out = self.p_qAttn(p_rnn_out, q_attn_out)
        c_attn_out = self.c_cAttn(c_rnn_out)

        preactivation = (self.p_c_interact(p_attn_out) + self.q_c_interact(q_attn_out))*c_attn_out
        preactivation = torch.sum(preactivation, dim = -1)
        return F.sigmoid(preactivation)

