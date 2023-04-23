import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from gan import fegAvgWithGan
from ganModels import discriminator,generator


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

# fedavg
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.8, help='client fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

# gan
parser.add_argument('-gan', '--use_gan', type=int, default=1, help='whether to use gan to generate virtual client data')
parser.add_argument('-tcp', '--training_client_percent', type=float, default=0.8, help='the percent of clients training in gan')
parser.add_argument('-gE', '--gan_epoch', type=int, default=30, help='gan model epoch')
parser.add_argument('-gB', '--gan_batchsize', type=int, default=5, help='gan model batch size')  # RAM AND CPU LIMITATION WARNING !  >=40
parser.add_argument('-dlr', '--discriminator_lr', type=float, default=0.002, help='learning rate of discriminator')
parser.add_argument('-glr', '--generator_lr', type=float, default=0.002, help='learning rate of generator')
parser.add_argument('-cp', '--client_percent', type=float, default=0.3, help='the percent of real clients in fedavg')



def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__=="__main__":  # 当模块直接运行时运行下面的代码，当程序被导入时代码块不运行
    args = parser.parse_args()  #解析参数
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']  #只看到需要使用的gpu
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # cuda是否可用

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if args['use_gan']:  # 是否使用gan生成虚假客户端
        #  在循环之前就创建对象，可以让下面的循环使用同一个网络
        noise_length = 128*11
        G_model = generator(noise_length)
        D_model = discriminator()

    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        net = torch.nn.DataParallel(net)
        G_model = torch.nn.DataParallel(G_model)
        D_model = torch.nn.DataParallel(D_model)
    net = net.to(dev)
    G_model = G_model.to(dev)
    D_model = D_model.to(dev)
    print("Create nets compete")

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))  # 如果使用所有客户端，则前者大，整个式子变成客户端的数量
                                                                           # 反之，则后者大，整个式子变为1

    global_parameters = {}
    for key, var in net.state_dict().items(): # 将net中的参数保存在字典中（是参数，不是训练梯度）
        # key,value格式例子：
        # conv1.weight 	 torch.Size([6, 3, 5, 5])
        # conv1.bias 	 torch.Size([6])
        # conv2.weight 	 torch.Size([16, 6, 5, 5])
        # conv2.bias 	 torch.Size([16])
        # fc1.weight 	 torch.Size([120, 400])
        # fc1.bias 	     torch.Size([120])
        # fc2.weight 	 torch.Size([84, 120])
        # fc2.bias 	     torch.Size([84])

        # .state_dict() 将每一层与它的对应参数建立映射关系
        # .item() 取出tensor中的值，变为Python的数据类型
        global_parameters[key] = var.clone()  # clone原来的参数，并且支持梯度回溯

    for comm in range(args['num_comm']): # 循环次数为通信次数
        print("communicate round {}".format(comm+1))

        order = np.random.permutation(args['num_of_clients'])  # np.random.permutation(数字)代表生成
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        #  如果上面num_in_comm 是1，则只选一个客户端，反之就选多个客户端

        sum_parameters = None
        for client in clients_in_comm:  # tqdm进度条（已删除），开始针对单一客户端
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
                                                                          #  客户端更新本地参数
            if sum_parameters is None:  # 第一次循环
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()  # 写入梯度
                for var in sum_parameters:
                    sum_parameters[var] = [sum_parameters[var]]
            else:  # 之后的循环
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var]+[local_parameters[var]] #  将参数添加进列表

        if args['use_gan']:  # 是否使用gan生成虚假客户端

            fake_client_num=args['client_percent']*args['num_of_clients']*args['cfraction']  # 这次训练用的clients数目 乘 假客户端的比例
            fake_clients=fegAvgWithGan(sum_parameters,
                                          G_model,
                                          D_model,
                                          args['training_client_percent'],
                                          args['gan_epoch'],
                                          args['gan_batchsize'],
                                          args['discriminator_lr'],
                                          args['generator_lr'],
                                          fake_client_num,
                                          dev)
            fake_parameters=fake_clients.main()
            for var1,var2 in zip(sum_parameters,fake_parameters):
                sum_parameters[var1]+=fake_parameters[var2]
            # 最终总数目为(真客户端数目*使用比例)*(1+假客户端的比例)
        else:  # 普通联邦学习
            fake_client_num=0

        for var in sum_parameters:
            sum_add=None
            for item in sum_parameters[var]:
                if sum_add==None:
                    sum_add=item
                else:
                    sum_add=sum_add+item
            global_parameters[var] = sum_add / (num_in_comm+fake_client_num)  # 求平均，FedAvg算法出现

        with torch.no_grad():  # 下面的操作不会被反向传播记录
            # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）
            # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():
            # 强制之后的内容不进行计算图构建。
            if (comm + 1) % args['val_freq'] == 0:  # 每隔多少次通信进行一次梯度的更新，更新到客户端的net上
                net.load_state_dict(global_parameters, strict=True) # 将预训练的参数权重加载到新的模型之中（将平均完成的模型分发到net中）
                # 当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合
                # 如果新构建的模型在层数上进行了部分微调，则上述代码就会报错：说key对应不上
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))


        #if (comm + 1) % args['save_freq'] == 0:  # 保存网络
        if True:  # 保存网络
            epoch_path = args['save_path'] + "/Num_comm=" + str(comm)
            test_mkdir(epoch_path)
            torch.save(net,os.path.join(epoch_path,"Main"))
            torch.save(G_model,os.path.join(epoch_path,"Generator"))
            torch.save(D_model,os.path.join(epoch_path,"Discriminator"))

#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         运行成功