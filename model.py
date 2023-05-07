import torch
import torch.nn as nn

class MegaSiameseModel(nn.Module):
    def __init__(self, bert1, bert2, out_dim):
        super().__init__()
        self.bert1 = bert1
        self.bert2 = bert2
        #del bert1.pooler, bert2.pooler
        self.emb1_dim = self.bert1.embeddings.word_embeddings.weight.shape[-1]
        self.emb2_dim = self.bert2.embeddings.word_embeddings.weight.shape[-1]
        self.out = nn.Linear(self.emb1_dim+self.emb2_dim, out_dim)
    
    def forward(self, batch):
        emb1 = self.bert1(batch['context']).last_hidden_state[:, 0, :]
        emb1 = torch.nn.functional.normalize(emb1)
        emb2 = self.bert1(batch['answer']).last_hidden_state[:, 0, :]
        emb2 = torch.nn.functional.normalize(emb2)
        x = torch.cat((emb1, emb2), dim=-1)
        return self.out(x)