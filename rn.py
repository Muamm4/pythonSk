import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os 


transform = transforms.ToTensor()

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28,128) # Camada de entrada, 784 neuronios que se liga a 128
        self.linear2 = nn.Linear(128,64) # camada interna 1, 128 neuronios que se liga a 64
        self.linear3 = nn.Linear(64,10) # camada interna 2, 64 neuronios que se liga a 10
        # para a camada de saida não é necessário definir nada pois precisamos pegar o output dea camada interna2
        
    def forward(self,x):
        x = F.relu(self.linear1(x)) # função de ativação de camada de entrada para camada interna 1
        x = F.relu(self.linear2(x)) # função de ativação da camada interna 1 para a camada interna 2
        x = self.linear3(x) # função de ativação da camada interna 2 para a camada de saida, nesse caso f(x) = x
        return F.log_softmax(x,dim=1)
    
def treino(modelo,trainloader,device):
    
    otimizador = optim.SGD(modelo.parameters(),lr=0.01,momentum=0.5)
    inicio = time();
    
    criterio = nn.NLLLoss()
    EPOCHS = 10
    modelo.train()
    
    for epoch in range(EPOCHS):
        perda_acumulada = 0
        
        for imagens, labels in trainloader:
            imagens = imagens.view(imagens.shape[0],-1)
            
            otimizador.zero_grad()
            
            output = modelo(imagens.to(device))
            perda_instantanea = criterio(output,labels.to(device))
            perda_instantanea.backward()
            otimizador.step()
            perda_acumulada += perda_instantanea.item()
        else:
            print('Epoch {} - Perda resultante: {}'.format(epoch+1,perda_acumulada/len(trainloader)))
        torch.save(modelo.state_dict(), './saves/save.pt')
    print('Tempo de treino (em minutos): {}'.format((time()-inicio)/60))

def validacao(modelo,testloader,device):
    conta_corretas, conta_todas = 0,0
    for imagens, labels in testloader:
        for i in range(len(labels)):
            img = imagens[i].view(1,784)
            # desativa o autograd para acelerar a validação
            with torch.no_grad():
                logps = modelo(img.to(device))
                
            ps = torch.exp(logps)
            probab = list(ps.cpu().numpy()[0])
            predito = probab.index(max(probab))
            classe = labels.numpy()[i]
            if predito == classe:
                conta_corretas += 1
            conta_todas += 1
    print("Porcentagem de acerto: {:.2f}%".format(100*conta_corretas/(conta_todas)))

save_path = './saves/save.pt'
modelo = Modelo()
if os.path.exists(save_path):
    modelo.load_state_dict(torch.load(save_path))
modelo.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)
# treino(modelo,trainloader,device)
validacao(modelo,testloader,device)