import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from ganModels import discriminator,generator


class dataset(Dataset):
    def __init__(self, parameters):
        self.parameters=parameters

    def flat(self,nums):  # 列表展开
        res = []
        for i in nums:
            if isinstance(i, list):
                res.extend(self.flat(i))
            else:
                res.append(i)
        return res

    def __getitem__(self, index):
        newTensor=[]
        for i in self.parameters[index]:
            newTensor.append(i.tolist())
        return torch.Tensor(self.flat(newTensor))

    def __len__(self):
        return len(self.parameters)

class fegAvgWithGan():
    def __init__(self,sum_parameters,G_model,D_model,training_client_percent,gan_epoch,gan_batchsize,discriminator_lr,generator_lr,fake_client_num,dev):
        self.sum_parameters=sum_parameters
        self.G_model=G_model
        self.D_model=D_model
        self.training_client_percent=training_client_percent
        self.gan_epoch=gan_epoch
        self.gan_batchsize=gan_batchsize
        self.discriminator_lr=discriminator_lr
        self.generator_lr=generator_lr
        self.fake_client_num=fake_client_num
        self.dev=dev

    def main(self):

        vars, parameters, sizes = self.ProcessData(self.sum_parameters)  # 把客户端化为列表，方便dataset处理
        random.shuffle(parameters)  # shuffle
        parameters = parameters[:int(len(parameters) * self.training_client_percent)]  # 使用部分client进行训练

        train_data = dataset(parameters)
        train_loader = DataLoader(train_data, batch_size=self.gan_batchsize, shuffle=True)

        fake_clients_list = self.CreateFakeClients(train_loader,self.G_model,self.D_model)

        return self.IntoDict(fake_clients_list,vars,sizes)


    def CreateFakeClients(self,train_loader,G_model,D_model,noise_length=128*11):

        fake_clients_list=[]


        loss = nn.BCELoss()

        G_opt = torch.optim.Adam(G_model.parameters(), self.generator_lr)
        D_opt = torch.optim.Adam(D_model.parameters(), self.discriminator_lr)

        for e in range(self.gan_epoch):
            print("GAN epoch {}".format(e))
            for i, data in enumerate(train_loader):
                real_batch = data.size(0)
                data=data.view(real_batch,-1).to(self.dev)

                #  define the one and zero label
                real_label = torch.ones(real_batch, 1).to(self.dev)
                fake_label = torch.zeros(real_batch, 1).to(self.dev)

                #  TRAIN DISCRIMINATOR
                #  compute loss of real image
                real_out = D_model(data)
                D_loss_real = loss(real_out, real_label)

                #  compute loss of fake image
                noise = torch.randn(real_batch,noise_length).to(self.dev)
                fake_client = G_model(noise)
                fake_out = D_model(fake_client)
                D_loss_fake = loss(fake_out, fake_label)

                #  add loss and optimize
                D_loss_total = D_loss_real + D_loss_fake
                D_opt.zero_grad()
                D_loss_total.backward()
                D_opt.step()

                #  TRAIN GENERATOR
                for i in range(3):
                    #  compute loss of fake img
                    noise = torch.randn(real_batch, noise_length).to(self.dev)
                    fake_client = G_model(noise)
                    fake_out = D_model(fake_client)
                    G_loss = loss(fake_out, real_label)

                    #  optimize
                    G_opt.zero_grad()
                    G_loss.backward()
                    G_opt.step()

            #if e%5==0 and e!=0:  # print loss
            if True:
                print("D:={} G:{}".format(D_loss_total, G_loss))


        #  generate fake clients
        for i in range(int(self.fake_client_num)):
            noise = torch.randn(1, noise_length).to(self.dev)
            fake_clients_list.append(G_model(noise).cpu())

        return fake_clients_list


    def IntoDict(self,fake_clients_list,vars,sizes):
        fake_clients_dict={}
        for i in vars:
            fake_clients_dict[i]=[]
        for each_client in fake_clients_list:  # 下面的是每个客户端
            each_client=each_client.detach().numpy().tolist()[0]
            for var,i in zip(fake_clients_dict,sizes):
                this_parameter=each_client[:np.prod(i)]  # 这个参数所占的列表
                each_client=each_client[np.prod(i):]  # 剩下的列表值
                fake_clients_dict[var].append(torch.Tensor(this_parameter).view(i).to(self.dev))  # 将这个列表转到tensor之后view成原来的形状，最后加到字典里
        return fake_clients_dict

    def ProcessData(self,dict):
        '''返回三个列表，一个是字典的索引，一个是数据，一个是数据的size'''
        vars=[]
        parameters=None
        sizes=[]
        for var in dict:
            vars.append(var)
            if parameters == None:
                parameters = [ [i] for i in dict[var] ]  # 列表里每个元素都是单元素列表
            else:
                for i in range(len(dict[var])):
                    parameters[i].append(dict[var][i])
        for i in parameters[0]:
            sizes.append(list(i.size()))
        return vars,parameters,sizes


