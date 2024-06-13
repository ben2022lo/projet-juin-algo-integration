import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scipy import signal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

TS = pd.read_csv("/content/acoustic_esc50.csv")
TS['horodate'] = pd.to_datetime(TS['horodate'])
TS['value'] = TS['value'].astype(float)
win = 16000
list_t = TS.horodate.values
list_value = TS.value.values


class Autoencoder(nn.Module):
    def __init__(self, input_dim=640, bottleneck_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SpectroAE:
    def __init__(self):
        self.win = 0
        self.list_t = []
        self.Autoencoder = Autoencoder
        self.sequences = 0 
        self.AE = 0    
        self.thresholds = []
        self.X_test_tensor = 0
        self.trained = 0
        self.input_dim = 0

    def analysis(self, list_t, list_value, win):
        
        if self.trained == 1:
            self.list_t = list_t
            self.sequences = np.reshape(list_value, (-1,self.win))
            spectro_detection = np.empty((self.sequences.shape[0], self.input_dim))
            for i in range(self.sequences.shape[0]):
                sampleFreqs, segmentTimes, sxx = signal.spectrogram(self.sequences[i], 16000)
                sxx_log = 10 * np.log10(sxx + 1e-15)
                spectro_detection[i,:] = sxx_log.flatten()
            self.X_test_tensor = torch.tensor(spectro_detection, dtype=torch.float)

        if self.trained == 0:
            self.win = win
            self.sequences = np.reshape(list_value, (-1,win))
            learn_size = int(0.75*self.sequences.shape[0])
            self.list_t = list_t[learn_size*win : self.sequences.shape[0]*win]

            sampleFreqs, segmentTimes, sxx = signal.spectrogram(self.sequences[0], 16000)
            self.input_dim = len(segmentTimes)*len(sampleFreqs)
            # spectro learn
            spectro_learn = np.empty((learn_size, sxx.shape[0]*sxx.shape[1]))
            for i in range(learn_size):
                sampleFreqs, segmentTimes, sxx = signal.spectrogram(self.sequences[i], 16000)
                sxx_log = 10 * np.log10(sxx + 1e-15)
                spectro_learn[i,:] = sxx_log.flatten()

            # spectro detection
            
            spectro_detection = np.empty((self.sequences.shape[0]-learn_size, sxx.shape[0]*sxx.shape[1]))
            for i in range(learn_size,self.sequences.shape[0]):
                sampleFreqs, segmentTimes, sxx = signal.spectrogram(self.sequences[i], 16000)
                sxx_log = 10 * np.log10(sxx + 1e-15)
                spectro_detection[i-learn_size,:] = sxx_log.flatten()

            X_train_tensor = torch.tensor(spectro_learn, dtype=torch.float)
            self.X_test_tensor = torch.tensor(spectro_detection, dtype=torch.float)

            # Create PyTorch DataLoader for training and testing data
            train_dataset = TensorDataset(X_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=learn_size, shuffle=True)
            # model
            self.AE = self.Autoencoder(input_dim=len(segmentTimes)*len(sampleFreqs), bottleneck_dim=64)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.AE.parameters(), lr=0.001)
            num_epochs = 100
            # Train the model
            for epoch in range(num_epochs):
                self.AE.train()
                train_loss = 0.0
                for data in train_loader:
                    inputs = data[0]
                    optimizer.zero_grad()
                    outputs = self.AE(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
            
            reconstruction_error = nn.functional.mse_loss(self.AE(inputs), inputs, reduction='none').mean(dim=1)
            plt.hist(reconstruction_error.cpu().detach().numpy(),bins=30)
            plt.show()
            self.thresholds = [torch.quantile(torch.abs(reconstruction_error), 0.9).item(),
                               torch.quantile(torch.abs(reconstruction_error), 0.95).item() ,
                               torch.quantile(torch.abs(reconstruction_error), 0.99).item()]
            self.trained = 1

    def detection(self):
        self.AE.eval()
        res = []    
        test_dataset = TensorDataset(self.X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            i = 0
            for data in test_loader:
                
                inputs = data[0]
                reconstruction_error = nn.functional.mse_loss(self.AE(inputs), inputs, reduction='none').mean(dim=1)
                if reconstruction_error < self.thresholds[0]:
                    ano = 0
                    confiance = 0
                else:
                    ano = 1
                    if reconstruction_error < self.thresholds[1]:
                        confiance = 0.5
                    elif reconstruction_error < self.thresholds[2]:
                        confiance = 0.75
                    else:
                        confiance = 1
                for j in range(self.win):
                    res.append([self.list_t[i*self.win+j], ano, confiance, reconstruction_error])
                i += 1
        return res

    def save_model(self, c_id, tm_id):
        # Save the state of the model
        torch.save({
            'model_state_dict': self.AE.state_dict(),
            'thresholds': self.thresholds,
            'win': self.win,
            'trained': self.trained,
            'input_dim': self.input_dim
        }, "SpectroAE_"+ c_id + "_" + tm_id + ".pth")

    def load_model(self, c_id, tm_id):
        # Load the state of the model
        checkpoint = torch.load("SpectroAE_"+ c_id + "_" + tm_id + ".pth")
        self.AE = self.Autoencoder(input_dim=checkpoint['input_dim'], bottleneck_dim=64)  
        self.AE.load_state_dict(checkpoint['model_state_dict'])
        self.thresholds = checkpoint['thresholds']
        self.win = checkpoint['win']
        self.trained = checkpoint['trained']
        self.input_dim = checkpoint['input_dim']
                
            
algo = SpectroAE()
algo.analysis(list_t, list_value, win)
res = algo.detection()
for k in range(20):
    print(res[k*win])
algo.save_model("/content/model.pth", 1, 1)

algo2 = SpectroAE()
algo2.load_model("/content/model.pth")

algo2.analysis(list_t[-320000:], list_value[-320000:], win)
res2 = algo2.detection()  

for k in range(20):
    print(res2[k*win])

