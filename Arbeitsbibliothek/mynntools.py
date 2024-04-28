import torch
import torch.nn as nn
import pytorch_lightning as pl
#from torchviz import make_dot
#from torch.utils.tensorboard import SummaryWriter
import numpy as np
from Arbeitsbibliothek.myshadow import ShadowShot, ShadowCollection
import random


# die Parameter des QAOA sind der Input
# die Bitstrings sind der Output (und stellen dar, was predicted werden soll)
# die Hidden Layers/Memories stellen QUbits dar, die vom RNN manipuliert werden sollen
# wie viele Schichten das RNN haben soll (Anzahl Zeitschritte = Anzahl Schichten in QAOA)
class SimpleRNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate = 0.001):
        super(SimpleRNN, self).__init__()
        self.lr = learning_rate
        self.hidden_size = hidden_size # Anzahl der Parameter pro Schicht (input) = Anzahl der Qubits (hidden) = Features pro Schicht
        output_size = hidden_size 
        self.input_dim = input_size
        # batch_first=True bedeutet, dass Input Tensor die Form (batch size, seq, feature size) sein muss
        # batch size = Anzahl der Beispiele; seq = Anzahl Zeitschritte bzw. Layers; feature size = Details pro Zeitschritt bzw. Inut_dim
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) # Lineare Layer, um RNN und Softmax zu "verbinden" / Fully Connected Layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initial hidden layer; auf 1/2 gesetzt, da Qubits-Initialzustände Null sind und durch Hadamar Gates zum Plus state werden (hat Erwartungswert 1/2); jedes Element im Batch kriegt seine eigene Hidden layer -> (Batch_size, Features)
        # siehe PyTorch docu: h0 = (D * num_layers=1, N=batch_size, H_in =hidden_size)
        #h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device) # Initialisierung mit Nullen
        h0 = torch.full((1, x.size(0), self.rnn.hidden_size), 0.5).to(x.device)  # Tensor mit (1, Anzahl Beispiele, Features) ; 1 nicht wichtig, nur da, um 2D Batch zu 3D Batch zumachen
        
        #print(time_steps)

        # Forward propagation von RNN
        out, _ = self.rnn(x, h0)

        # [:, -1, :] decodes the hidden state of the last time step
        out = out[:, -1, :]
        #out = self.fc(out)
        out = self.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        #loss = nn.CrossEntropyLoss()(y_hat, y)
        # reduction sum sorgt dafür, dass die Liste von BCE am Ende aufsummiert wird (siehe doku)
        loss = nn.BCELoss(reduction='sum')(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

# Parameter sollen hier als Listen oder Arrays übergeben werden  
def give_trained_model(X,Y, max_epoch = 10, learning_rate = 0.001):
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    print("NN Input", x)
    print("NN Soll Output", y)
    '''
    print("bei training")
    for i, j in zip(x, y):
        print(i, j)
    '''
    input_size = X.shape[2]
    hidden_size = Y.shape[1] # TODOO: guck, ob das richtig ist
    N = X.shape[0] # batch_size

    #print("Hallo", x.shape, y.shape)

    # Erstellen und Trainieren des Modells
    model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, learning_rate=learning_rate)
    trainer = pl.Trainer(max_epochs=max_epoch, accelerator='auto', devices='auto')
    trainer.fit(model, torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=N))

    # Tensorboard Summary
    writer = SummaryWriter('runs/test')
    writer.add_graph(model, torch.tensor(X, dtype=torch.float32))
    writer.close()

    return model, X

#beliebig viele Shadow Collections können reingegeben werden
def model_for_shadows_mean(*arg, max_epoch = 10):
    X = []
    Y = []
    for shadowCollect in arg:
        for shadow in shadowCollect.shadowList:
            X.append(shadow.get_parameters())
            Y.append(shadow.get_mean())
    X = np.array(X)
    Y = np.array(Y)
    return give_trained_model(X,Y, max_epoch=max_epoch)

def model_for_shadows(arg, max_epoch = 10, learning_rate = 0.001):
    X = []
    Y = []
    
    if not type(arg) is list:
        arg = [arg]

    for shadowCollect in arg:
        for shadow in shadowCollect.shadowList:
            parameters = shadow.get_parameters() # Konvention: beta1, gamma1, beta2, gamma2, ...
            for item in shadow.shadow:
                X.append(parameters)
                Y.append(item)
    X = np.array(X)
    Y = np.array(Y)
    return give_trained_model(X,Y, max_epoch=max_epoch, learning_rate=learning_rate) # Tuple wird hier zurück geben!!


# path kann relativ sein
def save_model(model, path):
    #torch.save(model.state_dict(), path)
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    #model.load_state_dict(torch.load(path))
    model.eval()

    return model

# Parameter Konvention: siehe Tensor X von model_for_shadows ODER dieselbe wie bei run_qaoa ("Batch" von Parameters)
def sample_shadow_from_NN(model, parameters, shots=512, weights = None):

    if not torch.is_tensor(parameters):
        # checks ob List/Array von Listen/Arrays (Listen und Arrays haben immer Attribute Len)
        if hasattr(parameters, "__len__") and hasattr(parameters[0], "__len__"):
            X = []
            for element in parameters:
                p = []
                old_convention = element # in QAOA ist Konvention: beta1, beta2,..., gamma1, gamma2,...
                new_convention = [] # neue Konvention: beta1, gamma1, beta2, gamma2, ...
                half = int(len(old_convention)/2)
                for i in range(half):
                    new_convention.append(old_convention[i])
                    new_convention.append(old_convention[i + half])
                for i in range(0, len(new_convention), 2):
                    p.append([new_convention[i], new_convention[i+1]])
                X.append(p)
            parameters = torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError('Keine Valide Konvention der Parameter')
    #print("beim samplen", parameters)
    shadowList = []
    distrobutionList = model(parameters)
    for distro in distrobutionList:
        #print(distro)
        shadow = []
        for i in range(shots):
            single_shot = []
            for prob in distro:
                if random.uniform(0, 1) > prob:
                    single_shot.append(1)
                else:
                    single_shot.append(0)
            shadow.append(single_shot)
        s = ShadowShot(shadow, None) # TODO: parameter richtig übergeben, anstatt None
        shadowList.append(s)
    measurement = ShadowCollection(shadowList, weights)
    return measurement
    

'''
# Beispiel-Daten
N = 64  # Batchgröße
T = 5   # Anzahl der Zeitschritte
F = 4   # Anzahl der Merkmale (Features)

# Erstellen Sie zufällige Daten mit den angegebenen Dimensionen
x = torch.randn(N, T, F)
y = torch.randint(0, F, (N,))

# Erstellen und Trainieren des Modells
model = SimpleRNN(input_size=F)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=N))

Q = model(torch.tensor(x, dtype=torch.float32))
make_dot(Q.mean(), params=dict(model.named_parameters()))
'''
