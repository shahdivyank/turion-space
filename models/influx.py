import influxdb_client
import pandas as pd
from influxdb_client.client.write_api import SYNCHRONOUS
from environs import env
import json

import torch
import torch.nn as nn
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

def predict(model, X):
    model.eval()
    batch_size = X.size(0)
    device = X.device

    h0 = torch.zeros(model.layer_dim, batch_size, model.hidden_dim).to(device)
    c0 = torch.zeros(model.layer_dim, batch_size, model.hidden_dim).to(device)

    with torch.no_grad():
        predicted, _, _ = model(X, h0, c0)
        predicted = predicted.detach().cpu().numpy()
    
    return predicted

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0 = None, c0 = None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn

def define_model(hidden, learning):
    model = LSTMModel(input_dim = 1, hidden_dim = hidden, layer_dim = 1, output_dim = 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning)

    return model, criterion, optimizer

def prepare(sequence):
    SEQ_LENGTH = sequence
    data = np.load("1.npy")
    testdata = np.load("1.npy")

    datapoints = len(data)

    data = data[:int(datapoints * 0.7)]
    testdata = testdata[int(datapoints * 0.7):]

    data = (data - np.mean(data)) / np.std(data)

    def create_sequences(data, testdata, seq_length):
        xs = []
        ys = []
        tests = []
        test_ys = []
        for i in range(len(data) - seq_length):
            x = data[i: (i + seq_length)] # 20 datapoints
            y = data[i + seq_length] # 21st datapoint
            test = testdata[i: (i + seq_length)]
            
            if i + seq_length < len(testdata):
                test = testdata[i: (i + seq_length)]
                test_y = testdata[i + seq_length]
                tests.append(test)
                test_ys.append(test_y)
            xs.append(x)
            ys.append(y)
        
        return np.array(xs), np.array(ys), np.array(tests), np.array(test_ys)

    X, y, TEST, TEST_y = create_sequences(data, testdata, SEQ_LENGTH)

    TRAIN_X = torch.tensor(X[:, :, None], dtype=torch.float32)
    TRAIN_Y = torch.tensor(y[:, None], dtype=torch.float32)

    TEST_X = torch.tensor(TEST[:, :, None], dtype=torch.float32)
    TEST_Y = torch.tensor(TEST_y[:, None], dtype=torch.float32)

    return TRAIN_X, TRAIN_Y, TEST_X, TEST_Y


def lstm():
    TRAIN_X, TRAIN_Y, TEST_X, _ = prepare(15)
    model, criterion, optimizer = define_model(200, 0.01)
    train(model, criterion, optimizer, TRAIN_X, TRAIN_Y, 400)

    return model, TEST_X

def env_setup():
    env.read_env()
    bucket = env("INFLUXDB_BUCKET")
    org = env("INFLUXDB_ORGANIZATION")
    token = env("INFLUXDB_TOKEN")

    return token, org, bucket

def parse_data():
    df = pd.read_csv("../data/100_thermal_data.csv")
    result = df.head()
    print(result)
    return df

def train(model, criterion, optimizer, TRAIN_X, TRAIN_Y, epochs):
    num_epochs = epochs
    h0, c0 = None, None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs, h0, c0 = model(TRAIN_X, h0, c0)

        loss = criterion(outputs, TRAIN_Y)
        loss.backward()
        optimizer.step()

        h0 = h0.detach()
        c0 = c0.detach()

    return h0, c0

def main():
    model, TEST_X = lstm()

    df = parse_data()
    index = 71
    test_index = 0

    token, org, bucket = env_setup()
    url="http://localhost:8086"

    client = influxdb_client.InfluxDBClient(
        url=url,
        token=token,
        org=org
    )

    write_api = client.write_api(write_options=SYNCHRONOUS)

    while test_index < 30:
        timestamp = df.iloc[index]

        prediction = predict(model, TEST_X[test_index].unsqueeze(0))[0][0]

        try:            
            point = (
                influxdb_client.Point("thermal_readings")
                .tag("sensor", "SENSOR1")
                .field("temperature", prediction)
                .time(timestamp)
            )
            write_api.write(bucket=bucket, org=org, record=point)

            index += 1
            test_index += 1

        except Exception as e:
            print("❌ ERROR:", e)

main()