import torch


class DatasetPredMarconi:
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :-1].values
        label = self.data.iloc[idx, -1:].values

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return {'data' : data, 'label' : label}