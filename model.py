import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pad_packed_sequence,
    pack_padded_sequence
)

# https://github.com/sharkmir1/Hierarchical-Attention-Network/blob/master/model.py


class EncoderWithAttention(nn.Module):

    input_features: int = 100
    hidden_dim: int = 100
    num_layers: int = 1
    att_dim: int = 200
    use_layer_norm: bool = True

    def __init__(self, **kwargs):
        super().__init__()
        for key, values in self.__class__.__annotations__.items():
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
        # Encoder
        self.gru = nn.GRU(self.input_features, self.hidden_dim, 
            num_layers=self.num_layers, batch_first=True, bidirectional=True,)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(2*self.hidden_dim, elementwise_affine=True)
        # Attention
        self.attention = nn.Linear(2*self.hidden_dim, self.att_dim)
        self.context_vector = nn.Linear(self.att_dim, 1, bias=False)

    def forward(self, packed_input):
        valid_bsz = packed_input.batch_sizes
        # Encoder Part
        packed_output, _ = self.gru(packed_input)
        if self.use_layer_norm:
            normed_output = self.layer_norm(packed_output.data)
        else:
            normed_output = packed_output
        # Attention Part
        att = torch.tanh(self.attention(normed_output.data)) # Get a_i
        att = self.context_vector(att).squeeze(1) # Get u_i^T @ u_outernal
        att = torch.exp(att -  att.max())
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)
        att_weights = att / torch.sum(att, dim=1, keepdim=True) # Get Softmax v
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True) # Unpack
        outputs = outputs * att_weights.unsqueeze(2)
        outputs = outputs.sum(dim=1) # Get hierarchical value
        return outputs, att_weights


class HierarchicalAttentionNetwork(nn.Module):

    def __init__(self, num_classes, vocab_size, emb_dim, Words={}, Sents={}, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.word_level_module = EncoderWithAttention(**Words)
        Sents['input_features'] = Words['hidden_dim'] * 2
        self.sent_level_module = EncoderWithAttention(**Sents)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*self.sent_level_module.hidden_dim, num_classes)

    def forward(self, docs, doc_lengths, sent_lengths):
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim=0, descending=True)
        docs = docs[doc_perm_idx]
        sent_lengths = sent_lengths[doc_perm_idx]

        packed_sents = pack_padded_sequence(
            docs, lengths=doc_lengths.tolist(), batch_first=True)
        packed_sent_lengths = pack_padded_sequence(
            sent_lengths, lengths=doc_lengths.tolist(), batch_first=True).data
        
        valid_bsz = packed_sents.batch_sizes

        sent_lengths, sent_perm_idx = packed_sent_lengths.sort(dim=0, descending=True)
        sents = packed_sents.data[sent_perm_idx]
        sents = self.embedding(sents)

        sents = self.dropout(sents)
        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)

        sents, word_att_weights = self.word_level_module(packed_words)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False) # Sorting
        sents = sents[sent_unperm_idx]
        word_att_weights = word_att_weights[sent_unperm_idx]

        sents = self.dropout(sents)
        packed_sents = PackedSequence(sents, valid_bsz)
        docs, sent_att_weights = self.sent_level_module(packed_sents)
        _, doc_unperm_idx = doc_perm_idx.sort(dim=0, descending=False) # Sorting
        docs = docs[doc_unperm_idx]

        word_att_weights = word_att_weights[doc_unperm_idx]
        sent_att_weights = sent_att_weights[doc_unperm_idx]

        scores = self.fc(docs)
        return scores, word_att_weights, sent_att_weights
