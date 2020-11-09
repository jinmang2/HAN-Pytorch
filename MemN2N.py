import torch
import torch.nn as nn


class AdjacentMemN2N(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hop, padtoken=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_hop = hop
        C = []
        for i in range(hop+1):
            C.append(nn.Embedding(vocab_size, embedding_dim, padding_idx=padtoken))
            C[i].weight.data.normal_(0, 0.1)
        self.C = nn.Sequential(*C)
            
    def process_single_layer(self, story, u, A, C):
        m_A = A(story)
        # why implement elementwise product??
        # query-key,value form?
        prob = torch.softmax(torch.sum(u * m_A, sum=2), dim=1)
        m_C = C(story)
        prob = prob.unsqueeze(2).expand_as(m_C)
        o_next = torch.sum(prob * m_C, dim=1)
        return u + o_next
                    
    def forward(self, story, q):
        u = self.C[0](q)
        for i in range(self.num_hop):
            A = self.C[i]
            C = self.C[i+1]
            u = self.process_single_layer(story, u, A, C)
        W = self.C[-1]
        a = torch.softmax(W(u))
        return a
