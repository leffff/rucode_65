import torch
import torch.nn as nn

class MegaSiameseModel(nn.Module):
    def __init__(self, bert, out_dim):
        super().__init__()
        self.bert = bert
        self.emb_dim = self.bert.embeddings.word_embeddings.weight.shape[-1]
        self.out = nn.Linear(self.emb_dim, out_dim)
    
    def forward(self, batch):
        emb = self.bert(batch['answer']).last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb)
        return self.out(emb)
    