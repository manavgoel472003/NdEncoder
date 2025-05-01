import torch
import torch.nn as nn
from ndlinear import NdLinear # pip install ndlinear


class Encoder(nn.Module):
    def __init__(self, in_channel, latent_channels):
        super().__init__()
        layers = [] 
        for i in range(len(latent_channels)):
            layers.append(nn.Conv2d(in_channel, latent_channels[i], kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(latent_channels[i]))
            in_channel = latent_channels[i]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(self, out_channel, latent_channels):
        super().__init__()
        layers = []
        latent_channels = latent_channels[::-1]
        for i in range(len(latent_channels)-1):
            layers.append(nn.ConvTranspose2d(latent_channels[i], latent_channels[i+1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(latent_channels[i+1]))

        layers.append(nn.ConvTranspose2d(latent_channels[i+1], out_channel, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(out_channel))        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
class AutoEncoder(nn.Module):
    def __init__(self, in_channel, latent_channels):
        super().__init__()
        self.encoder = Encoder(in_channel, latent_channels)
        self.decoder = Decoder(in_channel, latent_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec
    

class NdEncoder(nn.Module):
    def __init__(self, in_dim, latent_channels):
        super().__init__()
        x,y = in_dim[1:]
        layers = [nn.Conv2d(in_dim[0], latent_channels[0], kernel_size=4, stride=2, padding=1)]
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(latent_channels[0]))
        x//=2
        y//=2        
        for i in range(1, len(latent_channels)):
            layers.append(NdLinear((latent_channels[i-1],x,y), (latent_channels[i], x//2, y//2)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(latent_channels[i]))  
            x//=2
            y//=2         
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
class NdDecoder(nn.Module):
    def __init__(self, in_dim, latent_channels):
        super().__init__()
        x,y = in_dim[1:]
        l = len(latent_channels)
        x//=2**l
        y//=2**l
        layers = []
        latent_channels = latent_channels[::-1]
        for i in range(len(latent_channels)-1):
            layers.append(NdLinear((latent_channels[i],x,y), (latent_channels[i+1], x*2, y*2)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(latent_channels[i+1]))
            x*=2
            y*=2

        layers.append(nn.ConvTranspose2d(latent_channels[i+1], in_dim[0], kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(in_dim[0]))        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
class NdAutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_channels):
        super().__init__()
        self.encoder = NdEncoder(in_dim, latent_channels)
        self.decoder = NdDecoder(in_dim, latent_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec