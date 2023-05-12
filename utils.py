from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm.auto import tqdm


class TopG_Dataset(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 context_len=200,
                 answer_len=60,
                 train=True):
        self._df = df.copy()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.answer_len = answer_len
        self.train = train
        self.preprocess_df()

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx, :].copy()
        context, answer, label = row
        return (
            idx,
            context,
            answer,
            label
        )

    def preprocess_df(self):
        if self.train:
            label2class = {
                "people": 0,
                "ai": 1
            }
            self._df.label = self._df.label.map(label2class)
        else:
            self._df.label = 0
        self._df[["context", "answer"]] = self._df[["context", "answer"]].apply(
            lambda row: self.encode_sequence(row), result_type="expand", axis=1
        )

    def encode_sequence(self, row):
        context_seq = self.tokenizer.encode(
            text=row.context,
            add_special_tokens=True,
            max_length=self.context_len,
            truncation=True,
            padding='max_length',
            return_tensors="np",
        )
        answer_seq = self.tokenizer.encode(
            text=row.answer,
            add_special_tokens=True,
            max_length=self.answer_len,
            truncation=True,
            padding='max_length',
            return_tensors="np",
        )
        return context_seq, answer_seq


def topg_collate(x):
    idx, context, answer, label = zip(*x)
    idx = torch.tensor(idx)
    label = torch.tensor(label)
    context = torch.tensor(context).squeeze()
    answer = torch.tensor(answer).squeeze()
    return {
        "idx": idx,
        "context": context,
        "answer": answer,
        "label": label,
    }


def predict(model, 
            tokenizer, 
            test_df, 
            sub_df, 
            path_to_save,
            context_len=200,
            answer_len=60,
            batch_size=64,
            device=None):
    if device is None:
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_df["label"] = np.zeros(test_df.shape[0])
    test_dataset = TopG_Dataset(test_df, 
                           tokenizer, 
                           context_len, 
                           answer_len, 
                           train=False)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             collate_fn=topg_collate, 
                             shuffle=False)
    
    model = model.to(device)
    model.eval()
    preds = np.zeros(len(test_dataset))
    for batch in test_loader:
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            scores = model(batch)
        batch_preds = torch.argmax(scores, axis=1)
        preds[batch["idx"].detach().cpu().numpy()] = batch_preds.detach().cpu().numpy()
    
    class2label = {
    0: "people",
    1: "ai"
    }
    sub_df.label = preds
    sub_df.label = sub_df.label.map(class2label)

    sub_df.to_csv(path_to_save)



def predict_ensemble(models, 
            tokenizer, 
            test_df, 
            sub_df, 
            path_to_save,
            context_len=200,
            answer_len=60,
            batch_size=64,
            device=None):
    if device is None:
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_df["label"] = np.zeros(test_df.shape[0])
    test_dataset = TopG_Dataset(test_df, 
                           tokenizer, 
                           context_len, 
                           answer_len, 
                           train=False)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             collate_fn=topg_collate, 
                             shuffle=False)
    global_preds = []
    for model in tqdm(models):
        model = model.to(device)
        model.eval()
        preds = np.zeros(len(test_dataset))
        for batch in test_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                scores = model(batch)
            batch_preds = torch.sigmoid(scores)[:, 1]
            preds[batch["idx"].detach().cpu().numpy()] = batch_preds.detach().cpu().numpy()
        global_preds.append(preds)
    
    preds = ((sum(global_preds) / len(models)) > 0.5).astype(int)
    
    class2label = {
        0: "people",
        1: "ai"
    }
    sub_df.label = preds
    sub_df.label = sub_df.label.map(class2label)

    sub_df.to_csv(path_to_save)
