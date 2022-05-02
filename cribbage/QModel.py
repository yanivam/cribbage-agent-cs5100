import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os  # For saving model
import numpy as np

class LinearQNet(nn.Module):
    '''
    A two layer perceptron for learning how to play cribbage.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='perceptron_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            don = (done, )

        # Predicted Q-values given input state:
        pred = self.model(state)

        # Target Q-vals:
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Calculate the estimated reward as the observed reward plus what we estimate
                # the reward to be of the state we get to from taking the action we did:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # update reward
            target[idx][action[idx].item()] = Q_new

        # Then, adjust the model so as to minimize the difference between the Q-estimate
        # computed above (Q_new) and the predicted reward:
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()