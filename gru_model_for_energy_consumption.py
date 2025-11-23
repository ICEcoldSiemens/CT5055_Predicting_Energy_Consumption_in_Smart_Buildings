import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


def load_dataset(filepath, sequence_length=20, max_samples=10000):
    print("Loading the dataset...please wait...")
    df = pd.read_csv(filepath, sep=';', low_memory=False, na_values='?', parse_dates=[[0, 1]], dayfirst=True)
    df.rename(columns={'Date_Time': 'Timestamp'}, inplace=True)
    df.set_index('Timestamp', inplace=True)
    df = df.dropna()
    print("Dataset after cleaning:", df.shape)

    # Relevant input features and target factor to predict
    features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    target = 'Global_active_power'


    # Dataset is normalized
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_x.fit_transform(df[features])
    y = scaler_y.fit_transform(df[[target]])


    # Data sequences are created from dataset
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print("Total sequences for dataset:", X_seq.shape)


    # As dataset is large, the number of sequences is reduced to stop memory fault
    X_seq = X_seq[:max_samples]
    y_seq = y_seq[:max_samples]
    print("Using reduced sequences:", X_seq.shape)

    return X_seq, y_seq, scaler_y

# GRU Model represented as a class
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def reward(current_usage, preferred_usage):
    return -abs(current_usage - preferred_usage)

def which_action(predicted_usage, preferred_usage, actions, ep=0.1):
    if np.random.random() < ep:
        return np.random.choice(actions)
    difference = [abs(predicted_usage + a - preferred_usage) for a in actions]
    return actions[np.argmin(difference)]

def policy(gru_model, x, Y, scaler, actions):
    gru_model.eval()
    reward = 0
    preferred_usage = 0.5

    for i in range(len(x) - 1):
        state = torch.tensor(x[i:i + 1], dtype=torch.float32)
        prediction = gru_model(state).item()
        action = which_action(prediction, preferred_usage, actions)
        adjusted = prediction + action
        actual_val = Y[i+1][0]
        re = reward(adjusted,preferred_usage)
        reward += re

    print("Total reward:", reward)

# The model is provided the sample data sequences and goes through training
def train(X_seq, y_seq, scaler_y):
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.25, shuffle=False)

    device = torch.device("cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = GRUModel(input_size=X_train.shape[2], hidden_size=64, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("The model is being trained...")
    try:
        for train_cycle in range(40):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)

            # Reshapes output value and target values if required
            if output.shape != y_train.shape:
                print(f"Reshaping output from {output.shape} to {y_train.shape}")
                output = output.view(-1, 1)
                y_train = y_train.view(-1, 1)

            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            print(f"Training Cycle {train_cycle + 1}/40 - MSE: {loss.item():.4f}")

        # Improves training model + predict values
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy()
            y_actual = y_test.cpu().numpy()

        predictions_inverse = scaler_y.inverse_transform(predictions)
        actual_inverse = scaler_y.inverse_transform(y_actual)


        plt.figure(figsize=(12, 5))
        plt.plot(actual_inverse, label='Actual')
        plt.plot(predictions_inverse, label='Predicted')
        plt.title("Energy Consumption Forecasting (GRU)")
        plt.xlabel("Timesteps")
        plt.ylabel("Global Active Power (kW)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        actions = [-0.1,0,0.1]
        policy(model, X_test, y_test, scaler_y, actions)

    except Exception as e:
        print("Training Error:", e)


def main():
    filepath = 'household_power_consumption.csv'
    X_seq, y_seq, scaler_y = load_dataset(filepath)
    train(X_seq, y_seq, scaler_y)

if __name__ == '__main__':
    main()

