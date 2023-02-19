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
        self.dimension = 128
        self.max_token_len = max_token_len

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=self.dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.activation_classifier = nn.Linear(2 * self.dimension, 1)
        self.valence_classifier = nn.Linear(2 * self.dimension, 1)


    def forward(self, inputs):

        output, (h_n, c_n) = self.lstm(inputs)

        out_forward = output[range(len(output)), self.max_token_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_valence = self.valence_classifier(text_fea)
        text_valence = torch.squeeze(text_valence, 1)
        valance_out = torch.sigmoid(text_valence)

        text_activation = self.activation_classifier(text_fea)
        text_activation = torch.squeeze(text_activation, 1)
        activation_out = torch.sigmoid(text_activation)


        return (valance_out, activation_out)



if __name__ == '__main__':

    train_file = "./ser_traindev/train.json"
    dataset = SERDataset(train_file)
    device = torch.device('cpu')

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
            input = input.to(device)
            output = model(input)
            print(output)
            break
        break



