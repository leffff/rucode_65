import torch

class TopG_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer,context_len,answer_len):
        self._df = df.copy()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.answer_len = answer_len
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
        label2class = {
            "people": 0,
            "ai": 1
        }
        self._df.label = self._df.label.map(label2class)
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
        return (context_seq, answer_seq)


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