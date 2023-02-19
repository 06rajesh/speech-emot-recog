import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F


class SERDataset(Dataset):
    def __init__(self, json_path:str):
        self.target_file = json_path
        self.inputs = list()
        self.targets = list()
        self.max_token_len = 0
        self.feature_len = 0

        self._load_data()


    def _load_data(self):
        with open(self.target_file) as f:
            train_data = json.load(f)

        max_token_len = 0

        for k in train_data.keys():
            sample_data = train_data[k]
            features = sample_data['features']
            self.inputs.append(features)

            if len(features) > max_token_len:
                max_token_len = len(features)

            self.targets.append((sample_data['valence'], sample_data['activation']))

        self.max_token_len = max_token_len
        self.feature_len = len(self.inputs[0][1])
        print('All data loaded from file {} with a maximum length of {}'.format(self.target_file, max_token_len))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        target = self.targets[idx]
        sample = {
            'input': torch.tensor(self.inputs[idx]),
            'valence': torch.tensor(target[0]),
            'activation': torch.tensor(target[1])
        }

        return self._add_padding(sample)

    def _add_padding(self, sample):
        padded = torch.zeros(self.max_token_len, self.feature_len)

        input = sample['input']
        input_token_len = input.shape[0]
        padded[:input_token_len, :] = input

        return {
            'input': padded,
            'valence': sample['valence'],
            'activation': sample['activation'],
        }



class SERClassifier(nn.Module):
    def __init__(self, max_token_len:int, n_features:int):
        super(SERClassifier, self).__init__()

        kernel_size = 3
        self.conv = nn.Conv1d(in_channels=max_token_len, out_channels=200, kernel_size=(kernel_size,), padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.lstm = nn.LSTM(n_features, 25, batch_first=True)
        self.fc1 = nn.Linear(25, 15)


    def forward(self, inputs):
        # print(inputs.shape)
        # conv_out = F.relu(self.conv(inputs))
        # print(conv_out.shape)
        # pooled = self.maxpool(conv_out)
        # print(pooled.shape)

        lstm_out, (h_n, c_n) = self.lstm(inputs)
        print(lstm_out.shape)

        out = F.relu(self.fc1(lstm_out))
        print(out.shape)

        return



if __name__ == '__main__':

    train_file = "./ser_traindev/train.json"
    dataset = SERDataset(train_file)

    # train-test split of the dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # create batch
    batch_size = 4
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    model = SERClassifier(dataset.max_token_len, dataset.feature_len)

    num_epochs = 20
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        for i, sample in enumerate(train_loader):
            input = sample['input']
            output = model(input)

            break
        break



