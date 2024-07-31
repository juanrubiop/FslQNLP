# -*- coding: utf-8 -*-
"""nn_all_used_words.ipynb
This trains a Neural Network to get the parameters for a circuit given only all the words that will be used, in both training and testing.
to call this file call on cli python3 nn_all_used_words.py N_Qubits, where N_Qubits is the number of qubits you want to imnput
"""

import os
import pathlib
import sys
import torch
from torch import nn,matmul,kron,bmm
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from icecream import ic


from numpy import dot
from numpy.linalg import norm

#from google.colab import drive
#drive.mount('/content/drive')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

ic(f"Using {device} device")

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,labels,embeddings):
        self.labels=labels
        self.embeddings=embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return self.embeddings[idx],self.labels[idx]

preq_embeddings={}
resource_path="/home/jrubiope/FslQnlp/resources/embeddings/common_crawl/glove.42B.300d.txt"
#resource_path="resources/embeddings/common_crawl/glove.42B.300d.txt"
with open(resource_path, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "double")
        preq_embeddings[word] = vector

keys=['cooks', 'lady', 'fixes', 'bakes', 'program', 'breakfast', 'skillful', 'troubleshoots', 'supper', 'delightful', 'grills', 'delicious', 'guy', 'repairs', 'code', 'gentleman', 'dinner', 'someone', 'feast', 'sauce', 'boy', 'interesting', 'helpful', 'individual', 'man', 'software', 'runs', 'prepares', 'completes', 'useful', 'tool', 'adept', 'tasty', 'practical', 'flavorful', 'roasts', 'dexterous', 'woman', 'application', 'meal', 'noodles', 'soup', 'algorithm', 'executes', 'makes', 'person', 'snack', 'lunch', 'teenager', 'debugs', 'chicken', 'masterful']

TESTING=True

words=[]
embeddings=[]
for key, value in preq_embeddings.items():
    words.append(key)
    embeddings.append(value)

training_words=keys
training_embeddings=[preq_embeddings.get(key) for key in training_words]

dev_words=words[30:60]
dev_embeddings=embeddings[30:60]


def create_data(embeddings):
    training_data=[]
    training_labels=[]
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            cos=dot(embeddings[i][0], embeddings[j][0])/(norm(embeddings[i][0])*norm(embeddings[j][0]))
            training_labels.append(cos)
            new_embedding=np.append(embeddings[i],embeddings[j])

            training_data.append(new_embedding)

    training_data=torch.tensor(np.array(training_data),requires_grad=True,device=device)

    training_labels=torch.tensor(np.array(training_labels),requires_grad=True,device=device)
    training_labels_square=torch.square(training_labels)

    return training_data, training_labels_square

y_gate=torch.tensor([[0,-1j],[1j,0]],dtype=torch.complex128,requires_grad=True,device=device)
x_gate=torch.tensor([[0,1],[1,0]],dtype=torch.complex128,requires_grad=True,device=device)
z_gate=torch.tensor([[1,0],[0,-1]],dtype=torch.complex128,requires_grad=True,device=device)

def Id(N):
    gate=torch.eye(2**N,requires_grad=True,dtype=torch.complex128,device=device)    
    gate.retain_grad()
    return gate

def zero_bra(N):
    gate=torch.tensor([1.+0j if i==0 else 0 for i in range(2**N)],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def zero_ket(N):
    gate=torch.tensor([[1.+0j] if i==0 else [0] for i in range(2**N)],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def zero_1d_ket():
    gate=torch.tensor([[1.+0j],[0]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def zero_1d_bra():
    gate=torch.tensor([[1.+0j,0]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def one_1d_ket():
    gate=torch.tensor([[0],[1.+0j]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def one_1d_bra():
    gate=torch.tensor([[0,1.+0j]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def Ry(theta):
    gate = torch.linalg.matrix_exp(-0.5j*theta[:,:,None]*y_gate)
    return gate

def Rx(theta):
    gate = torch.linalg.matrix_exp(-0.5j*theta[:,:,None]*x_gate)
    return gate

def Rx(theta):
    gate = torch.linalg.matrix_exp(-0.5j*theta[:,:,None]*z_gate)
    return gate

CRx=lambda x: kron(Id(1),matmul(zero_1d_ket(),zero_1d_bra()))+kron(Rx(x),matmul(one_1d_ket(),one_1d_bra()))

training_data,training_labels=create_data(training_embeddings)
dev_data,dev_labels=create_data(dev_embeddings)

report_times=4
Batches=500
B_SIZE=round(training_data.shape[0]/Batches)

B=round(Batches/report_times)

training_object=CustomDataset(training_labels,training_data)
dev_object=CustomDataset(dev_labels,dev_data)

training_loader=DataLoader(training_object,batch_size=B_SIZE)
validation_loader=DataLoader(dev_object)

loss_fn = torch.nn.MSELoss()

N_Qubits=int(sys.argv[1])
N_PARAMS=3*N_Qubits-1

class PreQ(nn.Module):
    def __init__(self):
        super(PreQ,self).__init__()
        self.flatten = nn.Flatten(start_dim=0)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50,N_PARAMS)
        )
        self.double()

    def forward(self, x):
        #x = self.flatten(x)
        logits1=self.linear_relu_stack(x[:,0:300])
        logits2=self.linear_relu_stack(x[:,300:600])
        # ic(logits2.requires_grad,logits2.is_leaf)

        logits1_reshaped=torch.reshape(logits1,(logits1.shape[0],logits1.shape[1],1))
        logits2_reshaped=torch.reshape(logits2,(logits2.shape[0],logits2.shape[1],1))
        # ic(logits2_reshaped.requires_grad,logits2_reshaped.is_leaf)

        bra=torch.stack([zero_bra(N_Qubits)[None] for i in range(logits1_reshaped.shape[0])])
        ket=torch.stack([zero_ket(N_Qubits) for i in range(logits1_reshaped.shape[0])])

        # ic(bra.requires_grad,bra.is_leaf)
        # ic(ket.requires_grad,bra.is_leaf)

        circuit1=bmm(bra,self.get_quantum_state(parameters=logits2_reshaped).mH)
        circuit2=bmm(self.get_quantum_state(parameters=logits1_reshaped),ket)

        #ic(circuit1.requires_grad,circuit1.is_leaf)
        #ic(circuit2.requires_grad,circuit2.is_leaf)

        inner_product=self.flatten(bmm(circuit1,circuit2))
        fidelity=torch.square(torch.abs(inner_product))
        
        #output squared is the fidelity of t
        #ic(output.requires_grad,output.is_leaf)


        return fidelity

    def get_quantum_state(self,parameters):
        #first_layer=torch.stack( [   kron(  kron(Rx(parameters[:,0])[i],Rx(parameters[:,1])[i]) ,Rx(parameters[:,2])[i])  for i in range(parameters.shape[0])   ]   )
        #second_layer=torch.stack( [   kron(kron( Ry(parameters[:,3])[i],Ry(parameters[:,4])[i] ),Ry(parameters[:,5])[i] )  for i in range(parameters.shape[0])   ]  )
        # third_layer=kron(CRx(parameters[:,6]),Id(1))
        # fourth_layer=kron(Id(1),CRx(parameters[:,7]))
        #output=bmm(bmm(bmm(first_layer,second_layer),Cx_layers[0]),Cx_layers[1])

        first_layer=torch.stack([self.recursiveRx(i,parameters,N_Qubits-1) for i in range(parameters.shape[0])])
        second_layer=torch.stack([self.recursiveRy(i,parameters,2*N_Qubits-1) for i in range(parameters.shape[0])])

        rotation_layers=[first_layer,second_layer]

        Cx_layers=[  kron(Id(i),kron(CRx(parameters[:,2*N_Qubits+i]),Id(N_Qubits-i-2))) for i in range(N_Qubits-1) ]

        all_layers=rotation_layers+Cx_layers
        
        output=self.compose(all_layers)        

        # ic(first_layer.requires_grad,first_layer.is_leaf)
        # ic(second_layer.requires_grad,second_layer.is_leaf)
        # ic(third_layer.requires_grad,third_layer.is_leaf)
        # ic(fourth_layer.requires_grad,fourth_layer.is_leaf)

        return output

    def recursiveRx(self, i, parameters,counter):        
        if counter==1:
            return kron(Rx(parameters[:,counter])[i],Rx(parameters[:,1])[i])
        else:
            return kron(self.recursiveRx(i,parameters,counter-1),Rx(parameters[:,counter])[i])

    def recursiveRy(self, i, parameters,counter):        
        if counter==N_Qubits+1:
            return kron(Ry(parameters[:,counter-1])[i],Ry(parameters[:,counter])[i])
        else:
            return kron(self.recursiveRy(i,parameters,counter-1),Ry(parameters[:,counter])[i])

    def compose(self,layers):

        if len(layers)==2:
            return bmm(layers[0],layers[1])
        
        else:
            last_element=layers.pop()
            return bmm(self.compose(layers),last_element)
        

model = PreQ().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch

        output = model(inputs)


        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        loss.backward(retain_graph = True)
        # ic(loss.item())

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        mm=i%B_SIZE
        # if i % B_SIZE == B_SIZE-1:
        if i%B == B-1:
            last_loss = running_loss / B_SIZE # loss per batch
            ic('  batch {} loss: {}'.format(i + 1, last_loss*100))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
runs='/home/jrubiope/FslQnlp/runs/NN_outputs/AUW_{}_{}/Tensor_Board_Events/Prueba{}'.format(N_Qubits,N_PARAMS,timestamp)
#runs='runs/NN_outputs/AUW_{}_{}/Tensor_Board_Events/Prueba{}'.format(N_Qubits,N_PARAMS,timestamp)
path = pathlib.Path(runs)
path.mkdir(parents=True, exist_ok=True)
ic(runs)
model_path='/home/jrubiope/FslQnlp/runs/NN_outputs/AUW_{}_{}/Models'.format(N_Qubits,N_PARAMS)
#model_path='runs/NN_outputs/AUW_{}_{}/Models'.format(N_Qubits,N_PARAMS)
path = pathlib.Path(model_path)
path.mkdir(parents=True, exist_ok=True)

resources_path_model=f'/home/jrubiope/FslQnlp/resources/embeddings/NN/AUW_{N_Qubits}_{N_PARAMS}/Models'
path = pathlib.Path(resources_path_model)
path.mkdir(parents=True, exist_ok=True)


writer = SummaryWriter(runs)
epoch_number = 0
EPOCHS = 15
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    ep='EPOCH {}:'.format(epoch_number + 1)
    ic(ep)

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata

            voutputs = model(vinputs)

            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    calue='LOSS-- Train: {} Valid: {}'.format(avg_loss*100, avg_vloss*100)
    ic(value)

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model_path = model_path+'/best_model'
        best_model_resources=resources_path_model+'/best_model'
        #model_path = 'runs/NN_outputs/AUW_{}_{}/Models/best_model'.format(N_Qubits,N_PARAMS)
        torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), best_model_resources)
        


    epoch_number += 1


final_model_path = model_path+'/final_model'
torch.save(model.state_dict(), final_model_path)