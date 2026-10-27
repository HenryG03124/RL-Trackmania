import numpy as np
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self , in_channels , out_channels , stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size = 3 , stride = stride , padding = 1 , bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels , out_channels , kernel_size = 3 , stride = 1 , padding = 1 , bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels , out_channels , kernel_size = 1 , stride = stride , bias = False) ,
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self , x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 4 , out_channels = 64 , kernel_size = 8 , stride = 2 , padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 64 , out_channels = 64 , kernel_size = 4 , stride = 2 , padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 64 , out_channels = 128 , kernel_size = 4 , stride = 2 , padding = 0)
        self.conv4 = nn.Conv2d(in_channels = 128 , out_channels = 128 , kernel_size = 4 , stride = 2 , padding = 0)

        self.L1 = nn.Linear(137 , 256)
        self.L2 = nn.Linear(256 , 256)

        self.relu = nn.ReLU(inplace = True)

        self.mean = nn.Linear(256 , 3)
        self.log_std = nn.Linear(256 , 3)

    def forward(self , speed , gear , rpm , img , action1 , action2):
        feature = self.conv1(img)
        feature = self.relu(feature)
        feature = self.conv2(feature)
        feature = self.relu(feature)
        feature = self.conv3(feature)
        feature = self.relu(feature)
        feature = self.conv4(feature)
        feature = self.relu(feature)

        feature = torch.flatten(feature , start_dim = 1) #[B , 128]
        out = torch.cat((speed , gear , rpm , feature , action1 , action2) , dim = 1) #[B , 3 * 1 + 128 + 2 * 3 = 137]

        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)

        return self.mean(out) , torch.clip(self.log_std(out) , -20 , 2) #[B , 3(throttle , brake , steering)]

    def sample(self , speed , gear , rpm , img , action1 , action2 , test = False):
        mean , log_std = self.forward(speed , gear , rpm , img , action1 , action2)

        if test:
            return torch.tanh(mean) , None

        distribution = torch.distributions.Normal(mean , torch.exp(log_std))
        u = distribution.rsample()
        action = torch.tanh(u) #limit action to (-1 , 1) with gradient

        correction = 2 * (np.log(2) - u - torch.nn.functional.softplus(-2 * u))
        log_prob = (distribution.log_prob(u) - correction).sum(dim = -1)

        return action , log_prob #[B , 3] , [B]

class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 4 , out_channels = 64 , kernel_size = 8 , stride = 2 , padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 64 , out_channels = 64 , kernel_size = 4 , stride = 2 , padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 64 , out_channels = 128 , kernel_size = 4 , stride = 2 , padding = 0)
        self.conv4 = nn.Conv2d(in_channels = 128 , out_channels = 128 , kernel_size = 4 , stride = 2 , padding = 0)

        self.L1 = nn.Linear(140 , 256)
        self.L2 = nn.Linear(256 , 256)
        self.L3 = nn.Linear(256 , 1)

        self.relu = nn.ReLU(inplace = True)

    def forward(self , speed , gear , rpm , img , action1 , action2 , action):
        feature = self.conv1(img)
        feature = self.relu(feature)
        feature = self.conv2(feature)
        feature = self.relu(feature)
        feature = self.conv3(feature)
        feature = self.relu(feature)
        feature = self.conv4(feature)
        feature = self.relu(feature)

        feature = torch.flatten(feature , start_dim = 1) #[B , 128]
        out = torch.cat((speed , gear , rpm , feature , action1 , action2 , action) , dim = 1) #[B , 3 * 1 + 128 + 3 * 3 = 140]

        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)

        return out.squeeze(-1)