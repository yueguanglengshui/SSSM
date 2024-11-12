from __future__ import print_function
import os
from PIL import Image
import torchvision.models
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import time
import Resnet
from efficientnetv2_model import efficientnetv2_s as create_model
from model.model import Network_Wrapper
from utils import *
import sys
from torch.optim.lr_scheduler import LambdaLR
from model import *
import heapq
import statistics
# torch.cuda.set_device(1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 指定输出到文件的文件名
#output_file = 'output.txt'
def train(nb_epoch, batch_size, store_name, num_class,Semi_supervised,smalldataset,isshowimage,num_samples_per_class,train_image_total,resume=False, start_epoch=0, model_path=None, data_path = ''):
    # 将标准输出重定向到文件

    exp_dir = 'save_path/'+store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    output_file = 'save_path/' + store_name + '/infolog.txt'
    sys.stdout = open(output_file, 'w')

    # 获取当前时间的时间戳
    current_timestamp = time.time()

    # 将时间戳转换为结构化时间
    current_struct_time = time.localtime(current_timestamp)

    # 输出当前时间
    print("开始时间：", time.strftime("%Y-%m-%d %H:%M:%S", current_struct_time))

    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    logging.info(use_cuda)
    device = torch.device("cuda")
    print('==> Preparing data..')
    image_enhence_num=1
    train_path=data_path+'/train'
    # 指定随机种子
    # torch.manual_seed(0)
    # 每个类别随机选取多少倍num_samples_per_class与伪标签图像加入训练集
    train_pseudo_label_image_num = (train_image_total / num_class)/num_samples_per_class   #1350/45=30   1350/45=30/10=3
    train_pseudo_label_image_num=int(train_pseudo_label_image_num)
    # 每个类别最高产生多少倍与num_samples_per_class的伪标签图像数量
    pseudo_label_image_num = train_pseudo_label_image_num*4    #4分之1的概率选取
    pseudo_label_image_num=int(pseudo_label_image_num)
    #控制是否关闭伪标签生成，提前中断可以节约时间和降低伪标签的错误比例
    is_close_pseudo_label_produce=False
    is_used_all_choice_samples=False

    if Semi_supervised:
        full_dataset,labels_dataset,labeled_loader, unlabels_dataset,unlabeled_loader,val_dataset, val_loader, test_dataset,test_loader=loadData4(batch_size, data_path,num_samples_per_class,train_ratio=0.6, test_ratio=0.3, val_ratio=0.1,labeled_ratio=0.1)
    else:
        full_dataset,labels_dataset ,labeled_loader,val_dataset, val_loader, test_dataset,test_loader= loadData3(batch_size, data_path,train_ratio=0.6, test_ratio=0.3, val_ratio=0.1,labeled_ratio=0.1)
    #全部的索引
    unlabels_indices=unlabels_dataset.indices
    # image_data = {}
    image_data = [[] for _ in range(num_class)]
    labeled_data_origin = []
    labeled_data_origin_label = []
    unlabeled_data_origin=[]
    unlabeled_data_origin_label = []
    completion_image_list = []
    completion_label_list = []

    # pseudo_image_list = []
    # pseudo_label_list = []


    for images, targets in labeled_loader:

        for index, value in enumerate(targets):
            # 将伪标签添加到有标签数据集
            image = images[index]
            pseudo_label = targets[index]
            # image = images[index].unsqueeze(0)
            # pseudo_label = targets[index].unsqueeze(0)
            # assert image.size(0) == pseudo_label.size(0)
            # newdataset = torch.utils.data.TensorDataset(image, pseudo_label)
            # labeled_data_origin.append(newdataset)
            labeled_data_origin.append(image)
            labeled_data_origin_label.append(pseudo_label)

    del labels_dataset
    del labeled_loader
    completion_num=(batch_size-len(labeled_data_origin)%batch_size)
    if len(labeled_data_origin)>completion_num:
        completion_random_numbers = random.sample(range(0, len(labeled_data_origin)), completion_num)
        print("填补批次空挡数量：", completion_num)
        for i in completion_random_numbers:
            completion_image_list.append(labeled_data_origin[i])
            completion_label_list.append(labeled_data_origin_label[i])
    else:
        print("没有那么多数据，强行填补会报错")
    #进行批量补全，因为批量在很小的时候会对模型产生干扰
    labeled_dataset = torch.utils.data.TensorDataset(torch.stack(labeled_data_origin+completion_image_list), torch.stack(labeled_data_origin_label+completion_label_list))
    # 创建新的数据加载器
    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

    pseudo_dataset = torch.utils.data.TensorDataset(torch.stack(labeled_data_origin+completion_image_list),
                                                    torch.stack(labeled_data_origin_label+completion_label_list))
    # 创建新的数据加载器
    pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)
    if Semi_supervised:
        label_image_count = len(labeled_loader.dataset)
        print("labelImage count:", label_image_count)
        unlabel_image_count = len(unlabeled_loader.dataset)
        print("unlabelImage count:", unlabel_image_count)
        val_image_count = len(val_loader.dataset)
        print("valImage count:", val_image_count)
        test_image_count = len(test_loader.dataset)
        print("testImage count:", test_image_count)
    else:
        label_image_count = len(labeled_loader.dataset)
        print("labelImage count:", label_image_count)
        val_image_count = len(val_loader.dataset)
        print("valImage count:", val_image_count)
        test_image_count = len(test_loader.dataset)
        print("testImage count:", test_image_count)
        unlabel_image_count=0
    #

    #一半的未标记数据是多少张
    half_unlabel_image_count=unlabel_image_count/5*4
    #每个一半的未标记数据是多少张
    per_class=int(half_unlabel_image_count/num_class)    #每个类别如果伪标签数量到达这个数量，基本够用了，准备不生成伪标签了
    is_close_pseudo_label_produce_theta_num=int(per_class/4*3)*num_class   #关闭伪标签生成模块的指标

    print("每个类别如果伪标签数量到达这个数量就不再生成伪标签数据:",per_class)
    print("如果伪标签数量总数到达这个数量就关闭伪标签数据生成模块:", is_close_pseudo_label_produce_theta_num)
    #无标签索引，判断哪些已经用过了，好跳过

    index_lists = [0] * len(unlabels_indices)
    batch_size_num=int(unlabel_image_count/batch_size)
    #如果未标记数量太多，多久能跳出循环，循环次数是多少.加快训练时间
    loop_num=int(unlabel_image_count/batch_size)   #一共有多少个循环
    break_loop_num=int(10000/batch_size)   #到这个循环次数就跳出
    class_counts_origin = [num_samples_per_class] * num_class
    class_counts_hight = [0] * num_class   #高质量伪标签数量
    class_counts_hight_choice = [0] * num_class  # 被选中高质量伪标签数量
    class_counts_choice_samples = [0] * num_class    #当前选中的伪标签数量
    class_counts_low = [0] * num_class
    class_counts_choice = [0] * num_class
    class_counts_weight = [num_samples_per_class] * num_class  # 用于调控损失函数权重
    # pseudo_label_lists= [] * num_class
    pseudo_label_lists = [[] for _ in range(num_class)]
    pseudo_label_image_lists = [[] for _ in range(num_class)]
    pseudo_label_label_lists = [[] for _ in range(num_class)]
    unlabel_true_label_lists = [[] for _ in range(num_class)]
    pseudo_label_index_lists = [[] for _ in range(num_class)]  #用于存放选中的伪标签数据的下标，方便删除
    high_pseudo_label_index_lists = [[] for _ in range(num_class)] #用于存放选中的高质量伪标签数据的下标，方便删除
    pseudo_label_index_list = []
    pseudo_label_probability_lists = [[] for _ in range(num_class)]
    pseudo_label_pre_probability_lists = [[] for _ in range(num_class)]
    pseudo_label_mean_probability_lists=[0] * num_class
    #高质量的伪标签数据，可以长期持有
    high_quality_pseudo_image_list=[[] for _ in range(num_class)]
    high_quality_pseudo_label_list=[[] for _ in range(num_class)]
    #被选中的数据
    high_quality_pseudo_image_choicelist = []
    high_quality_pseudo_label_choicelist = []

    choice_pseudo_image=[]
    choice_pseudo_label=[]
    choice_pseudo_probability = []
    choice_true_label = []
    high_choice_true_label =[[] for _ in range(num_class)]
    # 错误伪标签数据的数量
    error_pseudo_num = 0
    true_pseudo_num = 0
    # class_counts = class_counts.to(device)
    weights = torch.Tensor([1 / count for count in class_counts_origin])
    # weights = weights.to(device)
    crop_image_num =1 # 每个聚类点裁几个part图片
    K =4  # 有几个聚类点
    crop_image_total = crop_image_num * K  # 一共有几个part图片
    channal=128
    channal1 = 256
    channal2 = 256
    channal3 = 256
    part_channal=256
    size1=16
    size2=16
    size3=16
    # resnet34 = torchvision.models.resnet34()
    # # state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    # state_dict = torch.load('./checkpoint/resnet34.pth')
    # resnet34.load_state_dict(state_dict)
    # # 修改模型的最后一层全连接层
    # # num_classes = 10  # 设置分类数量
    # resnet34.fc = nn.Linear(resnet34.fc.in_features, num_class)
    # partnet = partmodel(backbone=resnet34,num_class=num_class)
    # partnet = partmodel(num_class=num_class,channal=channal,K=K,crop_image_num=crop_image_num)
    # state_dict_part= torch.load('./save_path/CUB100/partmodel.pth')
    # partnet.load_state_dict(state_dict_part)
    # resnet50 = Resnet.resnet50(pretrained=True)

    efficientnetv2_backbone = create_model(num_classes=num_class)
    weights_dict = torch.load('./checkpoint/pre_efficientnetv2-s.pth')
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if efficientnetv2_backbone.state_dict()[k].numel() == v.numel()}
    print(efficientnetv2_backbone.load_state_dict(load_weights_dict, strict=False))

    net_teacher = Network_Wrapper(efficientnetv2_backbone, num_class,K=K,crop_image_num=crop_image_num)
    net_student = Network_Wrapper(efficientnetv2_backbone, num_class, K=K, crop_image_num=crop_image_num)
    # net_student=net_teacher
    #创建教师网络
    # 加载原始网络的参数到新的网络
    net_student.load_state_dict(net_teacher.state_dict())
    # net = Network_Wrapper(resnet50, num_class, K=K, crop_image_num=crop_image_num)

    # partnet=partmodel(num_class=num_class)


    # partnet=partmodel(efficientnetv2_backbone, num_class)
    # net = torch.load('./save_path/CUB100/model.pth')


    # state_dict= torch.load('./save_path/CUB100/model.pth')
    # net.load_state_dict(state_dict)

    #state_dict_part= torch.load('./save_path/CUB100/partmodel.pth')
    #partnet.load_state_dict(state_dict_part)

    # net.load_state_dict(copy.deepcopy(state_dict))
    # model_state_dict = torch.load('./save_path/CUB100/model.pth', map_location=torch.device('cpu'))
    # module = net.get_module()
    # module.load_state_dict(model_state_dict)
    # netp = torch.nn.DataParallel(net, device_ids=[0])



    net_teacher.to(device)
    net_student.to(device)
    # partnet.to(device)

    # CELoss = nn.CrossEntropyLoss(weights)

    # optimizer =optim.SGD(net.parameters(), lr=0.01)

    if num_samples_per_class==1:
        theta = 0.3
    elif num_samples_per_class==2:
        theta = 0.3
    elif num_samples_per_class==5:
        theta = 0.6
    elif num_samples_per_class==10:
        theta = 0.8
    else:
        theta = 0.9

    pseudo_label_theta = [theta] * num_class
    initial_lr = 0.05
    # optimizer1 = optim.SGD(net.parameters(), lr = initial_lr)
    optimizer = optim.SGD(net_teacher.parameters(), lr=initial_lr)
    # optimizer = optim.AdamW(net_teacher.parameters(), lr=0.05, weight_decay=0.01)
    # optimizer = optim.SGD(net_teacher.parameters(), lr=initial_lr)
    # scheduler_1 = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda epochs: 4/ (epochs + 4))
    # scheduler_1 = LambdaLR(optimizer, lr_lambda=lambda epochs: 0.002 if epochs >= 80 else 2 / (epochs + 2))
    max_val_acc = 0
    lambda_l2 = 0.01  # L1 正则化超参数，可以根据实际情况调整
    # 预测正确的数量
    error_num = 0
    # val_acc_com, val_loss = test(net, CELoss, batch_size, num_class, data_path + '/test', K, crop_image_num,channal)
    # isshowimage=True
    val_acc_com_current = 0
    loss_current=100

    # 数据增强 更新数据
    transform_train = transforms.Compose([
        transforms.Resize((310, 310)),
        # *chosen_augmentations,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((310, 310)),
    #     # *chosen_augmentations,
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(degrees=(-30, 30)),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    print("更新数据增强方式")
    labeled_loader.transform = transform_train
    print("初始化的学习率：", optimizer.defaults['lr'])
    # del labeled_data_origin
    labeled_data = 0
    # net_teacher.train()
    # net_student.train()
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        logging.info('\nEpoch: %d' % epoch)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        net_teacher.train()
        net_student.train()
        # partnet.train()
        train_loss0 = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0
        correct0 = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4=0
        correct5=0
        correct6 = 0
        correct7 = 0
        correct8 = 0
        correct9 = 0
        correct10 = 0
        correct11 = 0
        correct12 = 0
        correct13 = 0
        total = 0
        idx = 0


        print("Semi_supervised:", Semi_supervised)
        # 创建空的 TensorDataset 对象作为数据集
        # labeled_data=0    #用来表示每一次半监督中形成有标签数据的个数

        unlabeled_image = []  #用来存放新的无标签数据，后面会形成新的无标签数据加载器
        unlabeled_label = []
        unlabeled_data=[]
        theta_total=0
        # class_counts = [5] * num_class
        # weights = torch.Tensor([1 / count for count in class_counts])
        # weights = weights.to(device)
        # weights = torch.Tensor([1 / count for count in class_counts_choice])
        weights = weights.to(device)
        # CELoss = nn.CrossEntropyLoss(weights)
        CELoss = nn.CrossEntropyLoss()
        # CEloss=
        # CELoss=LabelSmoothingLoss(classes=num_class,smoothing=0.1)
        # 使用 SelfPacedLearningLoss 作为损失函数
        # CELoss = SelfPacedLearningLoss(num_class, threshold=0.5, start_value=0.0, end_value=1.0, step_size=0.1)
        if epoch <500:
            for batch_idx, (inputs, targets) in enumerate(labeled_loader):
                idx = batch_idx

                # if inputs.shape[0] < batch_size:
                #     continue
                # 将 tensor 转换为 PIL 图像
                # to_pil = transforms.ToPILImage()
                #
                # inputs_strong_pil = to_pil(inputs)
                # inputs_strong_pil = tensor_to_pil(inputs)
                # inputs_strong = transform_train(inputs_strong_pil)
                # for i in range(len(inputs)):
                #     inputs[i] = transform_test(inputs[i])
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)
                if isshowimage == True:
                    output_tensor = (inputs + 1) * 127.5
                    output_tensor = output_tensor[0]  # 取第一张图
                    imgcolors_uint8 = output_tensor.cpu().numpy()
                    imgcolors_uint8 = imgcolors_uint8.astype(np.uint8)
                    imgcolor_np = np.transpose(imgcolors_uint8, (1, 2, 0))
                    plt.imshow(imgcolor_np)
                    plt.show()

                weight_loss = []
                batches, channels, imgH, imgW = inputs.size()
                if batches == 1:
                    continue
                optimizer.zero_grad()
                x1_c, x2_c, x3_c, x_c_all, inputs_ATT= net_teacher(inputs,
                                                           index=0,
                                                           K=K,
                                                           crop_image_num=crop_image_num,
                                                           isBranch=True)

                loss = CELoss(x1_c, targets) + \
                       CELoss(x2_c, targets) + \
                       CELoss(x3_c, targets)
                # loss = ce_loss(x1_c, targets, reduction='mean') + \
                #        ce_loss(x2_c, targets, reduction='mean')+ \
                #        ce_loss(x3_c, targets, reduction='mean')
                part_classifier = x1_c + x2_c + x3_c
                loss.backward()
                optimizer.step()
                weight_loss.append(loss)
                if isshowimage == True:
                    output_tensor = (inputs_ATT + 1) * 127.5
                    output_tensor = output_tensor[0]  # 取第一张图
                    imgcolors_uint8 = output_tensor.cpu().numpy()
                    imgcolors_uint8 = imgcolors_uint8.astype(np.uint8)
                    imgcolor_np = np.transpose(imgcolors_uint8, (1, 2, 0))
                    plt.imshow(imgcolor_np)
                    plt.show()
                # 用第一分支，训练网络
                # e1
                optimizer.zero_grad()
                inputs1 = inputs_ATT

                part_classifier11,part_classifier11_1, part_classifier11_2,part_classifier11_3,inputsATT1 = net_teacher(inputs1, index=1,
                                                                             K=K,
                                                                             crop_image_num=crop_image_num,
                                                                             isBranch=True)


                # loss1 = CELoss(part_classifier11, targets) + \
                #         CELoss(part_classifier11_1, targets) + \
                #         CELoss(part_classifier11_2, targets) + \
                #         CELoss(part_classifier11_3, targets)
                # part_classifier_branch = part_classifier11 + part_classifier11_1 + part_classifier11_2 + part_classifier11_3
                loss1 =CELoss(part_classifier11_1, targets) + \
                        CELoss(part_classifier11_2, targets) + \
                        CELoss(part_classifier11_3, targets)
                # loss1 = ce_loss(part_classifier11_1, targets, reduction='mean') + \
                #         ce_loss(part_classifier11_2, targets, reduction='mean') + \
                #         ce_loss(part_classifier11_3, targets, reduction='mean')
                part_classifier_branch=part_classifier11_1+part_classifier11_2+part_classifier11_3
                weight_loss.append(loss1)
                loss2=loss1

                # l1_regularization = 0
                # for param in net.parameters():
                #     l1_regularization += torch.norm(param, 2)
                # loss1 += lambda_l2 * l1_regularization
                loss1.backward()
                optimizer.step()
                loss11_1 = CELoss(part_classifier11_1, targets)
                loss11_2 = CELoss(part_classifier11_2, targets)
                loss11_3 = CELoss(part_classifier11_3, targets)
                # loss11_1 =ce_loss(part_classifier11_1, targets, reduction='mean')
                # loss11_2 = ce_loss(part_classifier11_2, targets, reduction='mean')
                # loss11_3 = ce_loss(part_classifier11_3, targets, reduction='mean')



                weight_loss = torch.tensor(weight_loss, dtype=torch.float32)
                hidden_weight = (1 - (weight_loss - torch.min(weight_loss)) / (torch.max(weight_loss) - torch.min(weight_loss))) + torch.mean(weight_loss)
                output_all = part_classifier * hidden_weight[0] + part_classifier_branch * hidden_weight[1]

                predicted0 = F.softmax(part_classifier, dim=1)
                predicted1 = F.softmax(part_classifier11_1+part_classifier11_3, dim=1)
                predicted2 = F.softmax(part_classifier11_1, dim=1)
                predicted3 = F.softmax(part_classifier11_3, dim=1)
                predicted4 = F.softmax(part_classifier+part_classifier_branch, dim=1)
                predicted5 = F.softmax(output_all, dim=1)



                # print("阈值：",theta_total)
                predicted0 = torch.argmax(predicted0, dim=-1)
                predicted1 = torch.argmax(predicted1, dim=-1)
                predicted2 = torch.argmax(predicted2, dim=-1)
                predicted3 = torch.argmax(predicted3, dim=-1)
                predicted4 = torch.argmax(predicted4, dim=-1)
                predicted5 = torch.argmax(predicted5, dim=-1)

                predicted6 = voting(predicted0, predicted1, predicted2, predicted3, predicted4, predicted5,num_classes=num_class)
                # predicted6 = voting(vote_list)
                predicted6 = predicted6.to(device)
                # 进行投票


                total += targets.size(0)
                correct0 += predicted0.eq(targets.data).cpu().sum()
                correct1 += predicted1.eq(targets.data).cpu().sum()
                correct2 += predicted2.eq(targets.data).cpu().sum()
                correct3 += predicted3.eq(targets.data).cpu().sum()
                correct4 += predicted4.eq(targets.data).cpu().sum()
                correct5 += predicted5.eq(targets.data).cpu().sum()
                correct6 += predicted6.eq(targets.data).cpu().sum()
                # correct7 += predicted0.eq(targets.data).cpu().sum()
                # correct8 += predicted1.eq(targets.data).cpu().sum()
                # correct9 += predicted2.eq(targets.data).cpu().sum()
                # correct10 += predicted3.eq(targets.data).cpu().sum()
                # correct11 += predicted4.eq(targets.data).cpu().sum()
                # correct12 += predicted5.eq(targets.data).cpu().sum()
                # correct13 += predicted6.eq(targets.data).cpu().sum()

                train_loss0 += loss.item()
                train_loss1 += loss2.item()
                train_loss2 += loss11_1.item()
                train_loss3 += loss11_2.item()
                train_loss4 += loss11_3.item()
                train_loss5 += loss.item()+loss2.item()

                loss_current=train_loss2 / (batch_idx + 1)
                if batch_idx % 20 == 0:
                    logging.info('Step: %d | L0: %.5f | L1: %.5f | L2: %.5f | L3: %.5f | L4: %.5f' % (
                        batch_idx, train_loss0 / (batch_idx + 1), train_loss1 / (batch_idx + 1),
                        train_loss2 / (batch_idx + 1), train_loss3 / (batch_idx + 1),  train_loss4/ (batch_idx + 1),
                        ))
                    print('Step: %d | L0: %.5f | L1: %.5f | L2: %.5f | L3: %.5f | L4: %.5f' % (
                        batch_idx, train_loss0 / (batch_idx + 1), train_loss1 / (batch_idx + 1),
                        train_loss2 / (batch_idx + 1), train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1),
                    ))
                    logging.info(
                        'Step: %d | Acc0: %.3f%% | Acc1: %.3f%% | Acc2: %.3f%%| Acc3: %.3f%% | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%%|  total:%d' % (
                            batch_idx,
                            100. * float(correct0) / total, 100. * float(correct1) / total,100. * float(correct2) / total,
                            100. * float(correct3) / total, 100. * float(correct4) / total,100. * float(correct5) / total,
                            100. * float(correct5) / total,
                            total))
                    print(
                        'Step: %d | Acc0: %.3f%% | Acc1: %.3f%% | Acc2: %.3f%%| Acc3: %.3f%% | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%%|  total:%d' % (
                            batch_idx,
                            100. * float(correct0) / total, 100. * float(correct1) / total, 100. * float(correct2) / total,
                            100. * float(correct3) / total, 100. * float(correct4) / total, 100. * float(correct5) / total,
                            100. * float(correct5) / total,
                            total))
            logging.info('Step: %d | L0: %.5f | L1: %.5f | L2: %.5f | L3: %.5f | L4: %.5f' % (
                idx, train_loss0 / (batch_idx + 1), train_loss1 / (batch_idx + 1),
                train_loss2 / (batch_idx + 1), train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1),
            ))
            print('Step: %d | L0: %.5f | L1: %.5f | L2: %.5f | L3: %.5f | L4: %.5f' % (
                idx, train_loss0 / (batch_idx + 1), train_loss1 / (batch_idx + 1),
                train_loss2 / (batch_idx + 1), train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1),
            ))
            logging.info(
                'Step: %d | Acc0: %.3f%% | Acc1: %.3f%% | Acc2: %.3f%%| Acc3: %.3f%% | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%%|  total:%d' % (
                    idx,
                    100. * float(correct0) / total, 100. * float(correct1) / total, 100. * float(correct2) / total,
                    100. * float(correct3) / total, 100. * float(correct4) / total, 100. * float(correct5) / total,
                    100. * float(correct5) / total,
                    total))
            print(
                'Step: %d | Acc0: %.3f%% | Acc1: %.3f%% | Acc2: %.3f%%| Acc3: %.3f%% | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%%|  total:%d' % (
                    idx,
                    100. * float(correct0) / total, 100. * float(correct1) / total, 100. * float(correct2) / total,
                    100. * float(correct3) / total, 100. * float(correct4) / total, 100. * float(correct5) / total,
                    100. * float(correct5) / total,
                    total))
            train_acc = 100. * float(correct4) / total
            train_loss = train_loss5 / (idx + 1)
            with open(exp_dir + '/results_train.txt', 'a') as file:
                file.write(
                    'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_ATT: %.5f | Loss_concat: %.5f |\n' % (
                    epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                    train_loss4 / (idx + 1), train_loss5 / (idx + 1)))

        if epoch>75:
            # augmentations = [
            #     transforms.GaussianBlur(kernel_size=3),
            #     transforms.RandomResizedCrop(310, scale=(0.8, 1.0)),
            #     transforms.RandomErasing(p=0.5, scale=(0.1, 0.2), ratio=(0.5, 3.3), value='random'),
            #     transforms.Grayscale(num_output_channels=3),  # 灰度图像变换
            #     transforms.RandomAutocontrast(p=0.5),  # 图像的像素值分布均匀化
            #     transforms.RandomInvert(p=0.5),  # 随机反色处理
            #     transforms.RandomPosterize(2, p=0.5),
            #     transforms.RandomAdjustSharpness(2, p=0.5),
            #     transforms.RandomSolarize(threshold=192.0),  # 随机像素值取反
            #
            # ]
            # # 随机选取三种不重复的增强方法
            # chosen_augmentations = random.sample(augmentations, 4)
            transform_strong = transforms.Compose([
                # transforms.Resize((310, 310)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(-30, 30)),
                # *chosen_augmentations,
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            # KLDivLoss=nn.KLDivLoss()
            is_break = False
            # batch_idx=0
            # for batch_idx, (inputs, targets) in enumerate(pseudo_loader):
            for batch_idx,(pseudo_batch, labeled_batch) in enumerate(zip(pseudo_loader, labeled_loader)):
                pseudo_inputs, pseudo_targets = pseudo_batch
                labeled_inputs, labeled_targets = labeled_batch
                idx = batch_idx
                # batch_idx+=1
                weight_weak_loss = []
                weight_strong_loss = []
                inputs_weak = pseudo_inputs
                inputs_strong = pseudo_inputs
                break_loop_num=int(500/batch_size)
                # 对inputs_strong进行强增强
                # inputs_strong.to(torch.uint8)
                # inputs_strong = transform_strong(inputs_strong)
                # for i in range(len(inputs)):
                #     # img = transforms.ToPILImage()(inputs_strong[i])
                #     inputs_strong[i] = transform_strong(inputs_strong[i])
                if use_cuda:
                    inputs_weak = inputs_weak.to(device)
                    inputs_strong = inputs_strong.to(device)
                    labeled_inputs=labeled_inputs.to(device)
                    labeled_targets=labeled_targets.to(device)
                    pseudo_targets = pseudo_targets.to(device)
                if batch_idx == break_loop_num:
                    is_break = True
                    break
                batches, channels, imgH, imgW = inputs.size()
                if is_break == False:
            #         # 弱增强
                    net_teacher.train()
                    inputs_weak=torch.cat((inputs_weak,labeled_inputs),dim=0)
                    inputs_weak_target=torch.cat((pseudo_targets,labeled_targets),dim=0)
                    optimizer.zero_grad()
                    x1_c_weak, x2_c_weak, x3_c_weak, x_c_all_weak, inputs_ATT_weak = net_teacher(inputs_weak,
                                                                                                 index=0,
                                                                                                 K=K,
                                                                                                 crop_image_num=crop_image_num,
                                                                                                 isBranch=True)

                    part_classifier_weak = x1_c_weak + x2_c_weak + x3_c_weak
                    part_classifier_unlabel_loss = CELoss(x1_c_weak, inputs_weak_target) + \
                           CELoss(x2_c_weak, inputs_weak_target) + \
                           CELoss(x3_c_weak, inputs_weak_target)
                    # part_classifier_unlabel_loss= ce_loss(x1_c_weak, targets, reduction='mean') + \
                    #                                 ce_loss(x2_c_weak, targets, reduction='mean')+ \
                    #                                 ce_loss(x3_c_weak, targets, reduction='mean')


                    part_classifier_unlabel_loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()
                    part_classifier11_weak, part_classifier11_1_weak, part_classifier11_2_weak, part_classifier11_3_weak, inputsATT1_weak = net_teacher(
                                inputs_ATT_weak, index=1,
                                K=K,
                                crop_image_num=crop_image_num, isBranch=True)
                    part_classifier_branch_weak = part_classifier11_1_weak + part_classifier11_2_weak + part_classifier11_3_weak
                    # net_student.eval()

                    # with torch.no_grad():
                    #     x1_c_strong, x2_c_strong, x3_c_strong, x_c_all_strong, inputs_ATT_strong = net_student(
                    #         inputs_strong,
                    #         index=0,
                    #         K=K,
                    #         crop_image_num=crop_image_num,
                    #         isBranch=True)
                    #     part_classifier_strong = x1_c_strong + x2_c_strong + x3_c_strong
                    #     part_classifier11_strong, part_classifier11_1_strong, part_classifier11_2_strong, part_classifier11_3_strong, inputsATT1_strong = net_student(
                    #         inputs_ATT_strong, index=1,
                    #         K=K,
                    #         crop_image_num=crop_image_num, isBranch=True)
                    #     part_classifier_branch_strong = part_classifier11_1_strong + part_classifier11_2_strong + part_classifier11_3_strong
                    # part_classifier_branch_unlabel_loss = ce_loss(part_classifier11_1_weak, targets, reduction='mean') + \
                    #         ce_loss(part_classifier11_2_weak, targets, reduction='mean') + \
                    #         ce_loss(part_classifier11_3_weak, targets, reduction='mean')
                    part_classifier_branch_unlabel_loss = CELoss(part_classifier11_1_weak, inputs_weak_target) + \
                           CELoss(part_classifier11_2_weak, inputs_weak_target) + \
                           CELoss(part_classifier11_3_weak, inputs_weak_target)
                    # part_classifier_branch_unlabel_loss = CELoss(part_classifier_branch_weak, inputs_weak_target)
                    # consistency_loss = torch.nn.functional.mse_loss(torch.softmax(part_classifier_branch_strong, dim=1),
                    #                                         torch.softmax(part_classifier_branch_weak, dim=1))
                    # consistency_loss = torch.mean(torch.square(torch.softmax(part_classifier_branch_strong, dim=1) - torch.softmax(part_classifier_branch_weak, dim=1)))
                    # part_classifier_branch_unlabel_loss
                        # unlabel_loss =consistency_loss
                    # unlabel_loss = consistency_loss
                    # 步骤5：训练优化
                    # net_teacher.train()

                    part_classifier_branch_unlabel_loss.backward()
                    optimizer.step()
                    if batch_idx % 20 == 0:
                        logging.info(
                            'Step: %d |  total:%d' % (
                                idx,
                                idx*batch_size))
            logging.info(
                'Step: %d |  total:%d' % (
                    idx,
                    idx * batch_size))
        #         part_classifier_branch_weak = part_classifier11_1_weak + part_classifier11_2_weak + part_classifier11_3_weak
        #
        #         loss_all = KLDivLoss(part_classifier11_1_weak, part_classifier11_1_strong) + \
        #                        KLDivLoss(part_classifier11_2_weak, part_classifier11_2_strong) + \
        #                        KLDivLoss(part_classifier11_3_weak, part_classifier11_3_strong)
        #         # weight_strong_loss.append(loss2_strong)
        #         loss_all.backward()
        #
        #         optimizer.step()
        #
        #         logging.info(
        #             'Step: %d |  total:%d' % (
        #                 idx,
        #                 idx*batch_size))




        if epoch == 0:
            # val_acc_com, val_loss = test(net, test_loader,CELoss, batch_size, num_class, data_path + '/test', K, crop_image_num,channal)
            val_acc_com, val_loss = test(store_name,max_val_acc,net_teacher, test_loader, CELoss, batch_size, num_class, data_path, K,
                                         crop_image_num, channal)
            val_acc_com_current = val_acc_com
        # if epoch < 5 or epoch >= 100:
        if epoch > 30 and epoch % 20 == 0:
            # if epoch > 30 and epoch % 20 == 0 and loss_current < 0.03:
            # val_acc_com, val_loss =test(net, test_loader,CELoss, batch_size, num_class, data_path + '/test', K, crop_image_num,channal)
            # 获取当前时间戳
            start_time = time.time()
            print("测试时间")

            val_acc_com, val_loss = test(store_name,max_val_acc,net_teacher, test_loader, CELoss, batch_size, num_class, data_path, K,
                                         crop_image_num, channal)
            # 获取结束时间戳
            end_time = time.time()
            # 计算时间差
            elapsed_time = end_time - start_time
            print(f"测试经过的时间: {elapsed_time} 秒")

            val_acc_com_current=val_acc_com
            net_teacher.cpu()
            torch.save(net_teacher.state_dict(), './save_path/' + store_name + '/model.pth')
            # torch.save(efficientnetv2_backbone.state_dict(), './save_path/' + store_name + '/efficientnetv2_backbonemodel.pth')

            net_teacher.to(device)
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com

                net_student.load_state_dict(net_teacher.state_dict())
            # else:
                # net_teacher.load_state_dict(net_student.state_dict())
                # 定义EMA的衰减参数
                # decay = 0.99
                # print("EMA更新")
                # for ema_param, model_param in zip(net_teacher.parameters(), net_student.parameters()):
                #     net_teacher.data.mul_(decay).add_((1 - decay) * model_param.data)


                #
                # net.cpu()
                # torch.save(net.state_dict(), './save_path/' + store_name + '/model.pth')
                # net.to(device)
                # partnet.cpu()
                # torch.save(partnet.state_dict(), './save_path/' + store_name + '/partmodel.pth')
                # partnet.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('epoch %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc_com, val_loss))
        # else:
        #     net_teacher.cpu()
        #     torch.save(net_teacher.state_dict(), './save_path/' + store_name + '/model.pth')
        #     # torch.save(efficientnetv2_backbone.state_dict(),'./save_path/' + store_name + '/efficientnetv2_backbonemodel.pth')
        #
        #     net_teacher.to(device)
            # partnet.cpu()
            # torch.save(partnet.state_dict(), './save_path/' + store_name + '/partmodel.pth')
            # partnet.to(device)
        print('test_acc: %.3f | max_test_acc: %.3f '% (val_acc_com_current,max_val_acc))
        logging.info('test_acc: %.3f | max_test_acc: %.3f ' % (val_acc_com_current, max_val_acc))
        # scheduler_1.step()  #更新学习率
        # labeled_loader=labeled_loader_cope
        if smalldataset:
            Semi_supervised = False

        # 21 10  60   5   50  2   30    1  15
        # 21 10  50   5   40  2   25    1  10

        # 30 10  100  5   50  2   30    1  15
        # 30 10  90  5   40  2   25    1  10

        # 45 10  110  5   50  2   30    1  15
        # 45 10  100  5   40  2   25    1  10
        total_count1 = sum(class_counts_low)
        close_condition=num_class*90
        print("关闭伪标签生成条件为，图像数量超过：",close_condition)


        # if total_count<
        if(total_count1>close_condition):
            print("关闭伪标签生成")
            is_close_pseudo_label_produce=False

        if epoch == 61  or epoch == 69  or epoch == 75:
        # if epoch==61 or epoch==65 or epoch==69 or epoch==73 or epoch==75 :#or(epoch > 115 and epoch % 40 == 0):#or epoch==71 or epoch==72 or epoch==73 or epoch==74 or epoch==75 :#or(epoch > 115 and epoch % 20 == 0):
        # if (epoch > 75 and epoch % 20 == 0) or (epoch>40 and epoch <=60 and epoch%3==0) or  (epoch>140 and epoch <=152 and epoch%3==0):
            if Semi_supervised and is_close_pseudo_label_produce==False:
                # if Semi_supervised and is_close_pseudo_label_produce == False and loss_current < 0.03:
                is_break = False
                net_student.eval()
                net_teacher.eval()
                idx=0
                unlabeled_data_image = []
                unlabeled_data_label = []
                # del unlabeled_data
                current_produce_pseudo_num=0
                unlabel_image_count = len(unlabeled_loader.dataset)
                # unlabel_image_count = unlabel_count - labeled_data
                if unlabel_image_count > 0:

                    with torch.no_grad():

                        for batch_idx, (inputs, targets) in enumerate(unlabeled_loader):
                            idx = batch_idx
                            weight_weak_loss = []
                            weight_strong_loss = []
                            inputs_weak = inputs
                            inputs_strong = inputs
                            # 对inputs_strong进行强增强
                            # inputs_strong.to(torch.uint8)
                            # inputs_strong = transform_strong(inputs_strong)
                            # for i in range(len(inputs)):
                            #     # img = transforms.ToPILImage()(inputs_strong[i])
                            #     inputs_strong[i] = transform_strong(inputs_strong[i])
                            if use_cuda:
                                inputs_weak= inputs_weak.to(device)
                                inputs_strong = inputs_strong.to(device)
                                targets =targets.to(device)
                            if batch_idx==break_loop_num:
                                is_break=False
                            batches, channels, imgH, imgW = inputs.size()
                            if is_break==False:
                                #弱增强

                                x1_c_weak, x2_c_weak, x3_c_weak, x_c_all_weak, inputs_ATT_weak = net_teacher(inputs_weak,
                                                                            index=0,
                                                                            K=K,
                                                                            crop_image_num=crop_image_num,
                                                                            isBranch=True)


                                # part_classifier = x1_c + x2_c + x3_c + x_c_all
                                part_classifier_weak = x1_c_weak + x2_c_weak + x3_c_weak
                                loss1_weak = CELoss(x1_c_weak, targets) + \
                                       CELoss(x2_c_weak, targets) + \
                                       CELoss(x3_c_weak, targets)
                                # loss1_weak = ce_loss(x1_c_weak, targets, reduction='mean') + \
                                #              ce_loss(x2_c_weak, targets, reduction='mean') + \
                                #              ce_loss(x3_c_weak, targets, reduction='mean')
                                weight_weak_loss.append(loss1_weak)
                                # inputs = inputs_ATT_weak

                                part_classifier11_weak, part_classifier11_1_weak, part_classifier11_2_weak, part_classifier11_3_weak, inputsATT1_weak = net_teacher(
                                    inputs_ATT_weak, index=1,
                                    K=K,
                                    crop_image_num=crop_image_num, isBranch=True)

                                # part_classifier_branch = part_classifier11 + part_classifier11_1 + part_classifier11_2 + part_classifier11_3
                                part_classifier_branch_weak = part_classifier11_1_weak + part_classifier11_2_weak + part_classifier11_3_weak
                                loss2_weak = CELoss(part_classifier11_1_weak, targets) + \
                                        CELoss(part_classifier11_2_weak, targets) + \
                                        CELoss(part_classifier11_3_weak, targets)
                                # loss2_weak = ce_loss(part_classifier11_1_weak, targets, reduction='mean') + \
                                #              ce_loss(part_classifier11_2_weak, targets, reduction='mean') + \
                                #              ce_loss(part_classifier11_3_weak, targets, reduction='mean')
                                weight_weak_loss.append(loss2_weak)

                                weight_weak_loss = torch.tensor(weight_weak_loss, dtype=torch.float32)
                                hidden_weak_weight = (1 - (weight_weak_loss - torch.min(weight_weak_loss)) / (torch.max(weight_weak_loss) - torch.min(weight_weak_loss))) + torch.mean(weight_weak_loss)
                                output_weak_all = part_classifier_weak * hidden_weak_weight[0] + part_classifier_branch_weak * hidden_weak_weight[1]

                                #强增强
                                x1_c_strong, x2_c_strong, x3_c_strong, x_c_all_strong, inputs_ATT_strong = net_student(inputs_strong,
                                                                                                     index=0,
                                                                                                     K=K,
                                                                                                     crop_image_num=crop_image_num,
                                                                                                     isBranch=True)

                                # part_classifier = x1_c + x2_c + x3_c + x_c_all
                                part_classifier_strong = x1_c_strong + x2_c_strong + x3_c_strong
                                loss1_strong = CELoss(x1_c_strong, targets) + \
                                             CELoss(x2_c_strong, targets) + \
                                             CELoss(x3_c_strong, targets)
                                # loss1_strong = ce_loss(x1_c_strong, targets, reduction='mean') + \
                                #              ce_loss(x2_c_strong, targets, reduction='mean') + \
                                #              ce_loss(x3_c_strong, targets, reduction='mean')
                                weight_strong_loss.append(loss1_strong)
                                # inputs = inputs_ATT_weak

                                part_classifier11_strong, part_classifier11_1_strong, part_classifier11_2_strong, part_classifier11_3_strong, inputsATT1_strong = net_student(
                                    inputs_ATT_strong, index=1,
                                    K=K,
                                    crop_image_num=crop_image_num, isBranch=True)


                                # part_classifier_branch = part_classifier11 + part_classifier11_1 + part_classifier11_2 + part_classifier11_3
                                part_classifier_branch_strong =part_classifier11_1_strong + part_classifier11_2_strong + part_classifier11_3_strong
                                loss2_strong = CELoss(part_classifier11_1_strong, targets) + \
                                             CELoss(part_classifier11_2_strong, targets) + \
                                             CELoss(part_classifier11_3_strong, targets)
                                # loss2_strong = ce_loss(part_classifier11_1_strong, targets, reduction='mean') + \
                                #                ce_loss(part_classifier11_2_strong, targets, reduction='mean') + \
                                #                ce_loss(part_classifier11_3_strong, targets, reduction='mean')
                                weight_strong_loss.append(loss2_strong)
                                weight_strong_loss = torch.tensor(weight_strong_loss, dtype=torch.float32)
                                hidden_strong_weight = (1 - (weight_strong_loss - torch.min(weight_strong_loss)) / (torch.max(weight_strong_loss) - torch.min(weight_strong_loss))) + torch.mean(weight_strong_loss)
                                output_strong_all = part_classifier_strong * hidden_strong_weight[0] + part_classifier_branch_strong * hidden_strong_weight[1]

                                x1_c_weak_logits = F.softmax(x1_c_weak, dim=1)
                                x2_c_weak_logits = F.softmax(x2_c_weak, dim=1)
                                x3_c_weak_logits = F.softmax(x3_c_weak, dim=1)
                                x_c_all_weak_logits = F.softmax(x_c_all_weak, dim=1)
                                part_classifier_weak_logits= F.softmax(part_classifier_weak, dim=1)
                                part_classifier11_weak_logits = F.softmax(part_classifier11_2_weak+part_classifier11_3_weak, dim=1)
                                part_classifier11_1_weak_logits = F.softmax(part_classifier11_1_weak, dim=1)
                                part_classifier11_2_weak_logits = F.softmax(part_classifier11_2_weak, dim=1)
                                part_classifier11_3_weak_logits = F.softmax(part_classifier11_3_weak, dim=1)
                                part_classifier_branch_weak_logits = F.softmax(part_classifier_branch_weak, dim=1)
                                part_classifier_all_weak = part_classifier_weak + part_classifier_branch_weak
                                part_classifier_all_weak_logits = F.softmax(part_classifier_all_weak, dim=1)
                                part_classifier_all_weight_weak_logits = F.softmax(output_weak_all, dim=1)

                                #获取最大概率   和对应的下标
                                x1_c_weak_max_prob, x1_c_weak_index = torch.max(x1_c_weak_logits, dim=1)
                                x2_c_weak_max_prob, x2_c_weak_index = torch.max(x2_c_weak_logits, dim=1)
                                x3_c_weak_max_prob, x3_c_weak_index = torch.max(x3_c_weak_logits, dim=1)
                                x_c_all_weak_max_prob, x_c_all_weak_index = torch.max(x_c_all_weak_logits, dim=1)
                                part_classifier_weak_max_prob, part_classifier_weak_index = torch.max(part_classifier_weak_logits, dim=1)
                                part_classifier11_weak_max_prob, part_classifier11_weak_index = torch.max(part_classifier11_weak_logits, dim=1)
                                part_classifier11_1_weak_max_prob, part_classifier11_1_weak_index = torch.max(part_classifier11_1_weak_logits, dim=1)
                                part_classifier11_2_weak_max_prob, part_classifier11_2_weak_index = torch.max(part_classifier11_2_weak_logits, dim=1)
                                part_classifier11_3_weak_max_prob, part_classifier11_3_weak_index = torch.max(part_classifier11_3_weak_logits, dim=1)
                                part_classifier_branch_weak_max_prob, part_classifier_branch_weak_index = torch.max(part_classifier_branch_weak_logits, dim=1)
                                part_classifier_all_weak_max_prob, part_classifier_all_weak_index = torch.max(part_classifier_all_weak_logits, dim=1)
                                part_classifier_all_weight_weak_max_prob, part_classifier_all_weight_weak_index = torch.max(part_classifier_all_weight_weak_logits, dim=1)


                                x1_c_strong_logits = F.softmax(x1_c_strong, dim=1)
                                x2_c_strong_logits = F.softmax(x2_c_strong, dim=1)
                                x3_c_strong_logits = F.softmax(x3_c_strong, dim=1)
                                x_c_all_strong_logits = F.softmax(x_c_all_strong, dim=1)
                                part_classifier_strong_logits = F.softmax(part_classifier_strong, dim=1)
                                part_classifier11_strong_logits = F.softmax(part_classifier11_2_strong + part_classifier11_3_strong, dim=1)
                                part_classifier11_1_strong_logits = F.softmax(part_classifier11_1_strong, dim=1)
                                part_classifier11_2_strong_logits = F.softmax(part_classifier11_2_strong, dim=1)
                                part_classifier11_3_strong_logits = F.softmax(part_classifier11_3_strong, dim=1)
                                part_classifier_branch_strong_logits = F.softmax(part_classifier_branch_strong, dim=1)
                                part_classifier_all_strong = part_classifier_weak + part_classifier_branch_strong
                                part_classifier_all_strong_logits = F.softmax(part_classifier_all_strong, dim=1)
                                part_classifier_all_weight_strong_logits = F.softmax(output_strong_all, dim=1)
                                # 获取最大概率   和对应的下标
                                x1_c_strong_max_prob, x1_c_strong_index = torch.max(x1_c_strong_logits, dim=1)
                                x2_c_strong_max_prob, x2_c_strong_index = torch.max(x2_c_strong_logits, dim=1)
                                x3_c_strong_max_prob, x3_c_strong_index = torch.max(x3_c_strong_logits, dim=1)
                                x_c_all_strong_max_prob, x_c_all_strong_index = torch.max(x_c_all_strong_logits, dim=1)
                                part_classifier_strong_max_prob, part_classifier_strong_index = torch.max(part_classifier_strong_logits, dim=1)
                                part_classifier11_strong_max_prob, part_classifier11_strong_index = torch.max(part_classifier11_strong_logits, dim=1)
                                part_classifier11_1_strong_max_prob, part_classifier11_1_strong_index = torch.max(part_classifier11_1_strong_logits, dim=1)
                                part_classifier11_2_strong_max_prob, part_classifier11_2_strong_index = torch.max(part_classifier11_2_strong_logits, dim=1)
                                part_classifier11_3_strong_max_prob, part_classifier11_3_strong_index = torch.max(part_classifier11_3_strong_logits, dim=1)
                                part_classifier_branch_strong_max_prob, part_classifier_branch_strong_index = torch.max(part_classifier_branch_strong_logits, dim=1)
                                part_classifier_all_strong_max_prob, part_classifier_all_strong_index = torch.max(part_classifier_all_strong_logits, dim=1)
                                part_classifier_all_weight_strong_max_prob, part_classifier_all_weight_strong_index = torch.max(part_classifier_all_weight_strong_logits, dim=1)

                                # predicted_ATT_part_logit=predicted_ATT_part_logits
                                # # 判断伪标签的条件来决定是否添加到有标签数据集
                                # part_classifier_max_prob, part_classifier_xia = torch.max(part_classifier_logits, dim=1)
                                # part_classifier11_max_prob, part_classifier11_part_xia = torch.max(part_classifier11_logits, dim=1)
                                # max_prob, predicted_ATT_part = torch.max(predicted_ATT_part_logit, dim=1)  # 对应的最大概率
                                # 将伪标签添加到有标签数据集
                                inputs = inputs.cpu()
                                targets=targets.cpu()
                                x1_c_weak_index = x1_c_weak_index.cpu()
                                x2_c_weak_index = x2_c_weak_index.cpu()
                                x3_c_weak_index = x3_c_weak_index.cpu()
                                x_c_all_weak_index = x_c_all_weak_index.cpu()
                                part_classifier_weak_index=part_classifier_weak_index.cpu()
                                part_classifier11_weak_index = part_classifier11_weak_index.cpu()
                                part_classifier11_1_weak_index = part_classifier11_1_weak_index.cpu()
                                part_classifier11_2_weak_index = part_classifier11_2_weak_index.cpu()
                                part_classifier11_3_weak_index = part_classifier11_3_weak_index.cpu()
                                part_classifier_branch_weak_index = part_classifier_branch_weak_index.cpu()
                                part_classifier_all_weak_index = part_classifier_all_weak_index.cpu()
                                part_classifier_all_weight_weak_index = part_classifier_all_weight_weak_index.cpu()

                                x1_c_strong_index = x1_c_strong_index.cpu()
                                x2_c_strong_index = x2_c_strong_index.cpu()
                                x3_c_strong_index = x3_c_strong_index.cpu()
                                x_c_all_strong_index = x_c_all_strong_index.cpu()
                                part_classifier_strong_index = part_classifier_strong_index.cpu()
                                part_classifier11_strong_index = part_classifier11_strong_index.cpu()
                                part_classifier11_1_strong_index = part_classifier11_1_strong_index.cpu()
                                part_classifier11_2_strong_index = part_classifier11_2_strong_index.cpu()
                                part_classifier11_3_strong_index = part_classifier11_3_strong_index.cpu()
                                part_classifier_branch_strong_index = part_classifier_branch_strong_index.cpu()
                                part_classifier_all_strong_index = part_classifier_all_strong_index.cpu()
                                part_classifier_all_weight_strong_index = part_classifier_all_weight_strong_index.cpu()
                                #1张图片
                                # print(index_lists[batch_idx])
                                # print(targets)
                                for index in range(batches):#and x_c_all_max_prob[index]>=theta   and part_classifier11_max_prob[index]>=theta
                                    #如果等于1说明已经是伪标签了，就不要运行
                                    # print(index_lists[batch_idx])

                                    if index_lists[batch_idx*batch_size+index]==0:
                                        pseudo_label_list = []
                                        if x1_c_weak_index[index] == x1_c_strong_index[index]:
                                            pseudo_label_list.append(x1_c_weak_index[index])
                                        if x2_c_weak_index[index] == x2_c_strong_index[index]:
                                            pseudo_label_list.append(x2_c_weak_index[index])
                                        if x3_c_weak_index[index] == x3_c_strong_index[index]:
                                            pseudo_label_list.append(x3_c_weak_index[index])
                                        # # pseudo_label_list.append(x_c_all_index[index])
                                        if part_classifier_weak_index[index]==part_classifier_strong_index[index]:
                                            pseudo_label_list.append(part_classifier_weak_index[index])
                                        # # pseudo_label_list.append(part_classifier11_index[index])
                                        if part_classifier11_1_weak_index[index] == part_classifier11_1_strong_index[index]:
                                            pseudo_label_list.append(part_classifier11_1_weak_index[index])
                                        if part_classifier11_2_weak_index[index] == part_classifier11_2_strong_index[index]:
                                            pseudo_label_list.append(part_classifier11_2_weak_index[index])
                                        if part_classifier11_3_weak_index[index] == part_classifier11_3_strong_index[index]:
                                            pseudo_label_list.append(part_classifier11_3_weak_index[index])
                                        if part_classifier_branch_weak_index[index] == part_classifier_branch_strong_index[index]:
                                            pseudo_label_list.append(part_classifier_branch_weak_index[index])
                                        if part_classifier_all_weak_index[index] == part_classifier_all_strong_index[index]:
                                            pseudo_label_list.append(part_classifier_all_weak_index[index])
                                        if part_classifier_all_weight_weak_index[index] == part_classifier_all_weight_strong_index[index]:
                                            pseudo_label_list.append(part_classifier_all_weight_weak_index[index])
                                        pseudo_label_list.append(10000)
                                        pseudo_label = find_most_common(pseudo_label_list)
                                        count = pseudo_label_list.count(pseudo_label)
                                        if count >= 8 and class_counts_low[pseudo_label]<=100:
                                        # if count >= 8:
                                        # if part_classifier_all_weight_weak_index[index] == part_classifier_all_weight_strong_index[index]:
                                            # if x1_c_weak_max_prob[index]>=pseudo_label_theta[pseudo_label] and x2_c_weak_max_prob[index]>=pseudo_label_theta[pseudo_label] and x3_c_weak_max_prob[index]>=pseudo_label_theta[pseudo_label]  and part_classifier_weak_max_prob[index]>=pseudo_label_theta[pseudo_label]  and part_classifier11_1_weak_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier11_2_weak_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier11_3_weak_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier_branch_weak_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier_all_weak_max_prob[index]>=pseudo_label_theta[pseudo_label]:
                                            if part_classifier_weak_max_prob[index] >= pseudo_label_theta[pseudo_label] and part_classifier11_1_weak_max_prob[index] >= pseudo_label_theta[pseudo_label] and part_classifier11_2_weak_max_prob[index] >= pseudo_label_theta[pseudo_label] and part_classifier11_3_weak_max_prob[index] >=pseudo_label_theta[pseudo_label] and part_classifier_branch_weak_max_prob[index] >= pseudo_label_theta[pseudo_label] and part_classifier_all_weak_max_prob[index] >= pseudo_label_theta[pseudo_label]:
                                            # pseudo_label=part_classifier_all_weight_weak_index[index]
                                            # if part_classifier_all_weight_weak_max_prob[index] >= pseudo_label_theta[pseudo_label] and part_classifier_all_weight_strong_max_prob[index] >= pseudo_label_theta[pseudo_label]:
                                                image = inputs[index]
                                                # 已经生成伪标签，下次就不经过了
                                                # index_lists[batch_idx][index] = 1
                                                index_lists[batch_idx * batch_size + index]=1
                                                classifier = int(pseudo_label.item())
                                                pseudo_label_image_lists[classifier].append(image)
                                                pseudo_label_label_lists[classifier].append(pseudo_label)
                                                # 存放真实标签，方便查看是否降低了错误率
                                                unlabel_true_label_lists[classifier].append(targets[index])
                                                labeled_data += 1
                                                current_produce_pseudo_num+=1
                                                # print("true target:", targets[index])
                                                # print("pseudo_label :", pseudo_label)
                                                # inputs= inputs.to(device)
                                                class_counts_low[pseudo_label] = class_counts_low[pseudo_label] + 1

                                                # 添加概率分布
                                                pseudo_label_probability_lists[pseudo_label].append(
                                                    part_classifier11_3_weak_max_prob[index].item())
                                                pseudo_label_pre_probability_lists[pseudo_label].append(
                                                    x2_c_weak_max_prob[index].item())
                                                if targets[index] != pseudo_label:
                                                    error_num += 1
                                        # else:
                                        #     index_lists[idx][index] = 0
                                        # if x1_c_max_prob[index]>=pseudo_label_theta[pseudo_label] and x2_c_max_prob[index]>=pseudo_label_theta[pseudo_label] and x3_c_max_prob[index]>=pseudo_label_theta[pseudo_label]  and part_classifier_max_prob[index]>=pseudo_label_theta[pseudo_label]  and part_classifier11_1_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier11_2_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier11_3_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier_branch_max_prob[index]>=pseudo_label_theta[pseudo_label] and part_classifier_all_max_prob[index]>=pseudo_label_theta[pseudo_label]:
                                        # # if True:
                                        #     # 将伪标签添加到有标签数据集
                                        #
                                        #     if count >= 8:
                                        #         # newdataset = torch.utils.data.TensorDataset(image, pseudo_label)
                                        #         # new_labeled_data_origin.append(newdataset)
                                        #         # labeled_data_origin.append(newdataset)
                                        #         # labeled_data_origin.append(image)
                                        #         # labeled_data_origin_label.append(pseudo_label)
                                        #         #已经生成伪标签，下次就不经过了
                                        #         index_lists[batch_idx][index] = 1
                                        #
                                        #         classifier = int(pseudo_label.item())
                                        #         pseudo_label_image_lists[classifier].append(image)
                                        #         pseudo_label_label_lists[classifier].append(pseudo_label)
                                        #         #存放真实标签，方便查看是否降低了错误率
                                        #         unlabel_true_label_lists[classifier].append(targets[index])
                                        #         labeled_data+=1
                                        #         # print("true target:", targets[index])
                                        #         # print("pseudo_label :", pseudo_label)
                                        #         # inputs= inputs.to(device)
                                        #         class_counts_low[pseudo_label]=class_counts_low[pseudo_label]+1
                                        #
                                        #         #添加概率分布
                                        #         pseudo_label_probability_lists[pseudo_label].append(part_classifier_branch_max_prob[index].item())
                                        #
                                        #         if targets[index]!=pseudo_label:
                                        #             error_num+=1

                                            # else:
                                            #     image = inputs[index]
                                            #     pseudo_label = targets[index]
                                            #     unlabeled_data_image.append(image)
                                            #     unlabeled_data_label.append(pseudo_label)
                                                # index_list[idx][index]=0
                                        # else:
                                        #     image = inputs[index]
                                        #     pseudo_label = targets[index]
                                        #     unlabeled_data_image.append(image)
                                        #     unlabeled_data_label.append(pseudo_label)


                            # else:
                            #     inputs = inputs.cpu()
                            #     for index in range(batches):
                            #         image = inputs[index]
                            #         pseudo_label = targets[index]
                            #         unlabeled_data_image.append(image)
                            #         unlabeled_data_label.append(pseudo_label)
                                    # index_list[idx][index]=0
                                    # assert image.size(0) == pseudo_label.size(0)
                                    # unlabeled_image.append(image)
                                    # unlabeled_label.append(pseudo_label)
                                    # newdataset = torch.utils.data.TensorDataset(image, pseudo_label)
                                    # unlabeled_data.append(newdataset)
                           #当到达2000张图片时

                           #当达到4000张图片时

                           #当到达6000图片

                           #当到达8000张图片



                            if idx % 20 == 0:
                                label_image_count = len(labeled_loader.dataset)
                                unlabel_count = len(unlabeled_loader.dataset)
                                # print("labelImage count:", label_image_count+len(labeled_data))
                                # print("unlabelImage count:", len(unlabeled_data))
                                logging.info(
                                    'batch_idx: %d | labeled_loader number: %.3f | labeled_data number: %.5f | unlabeled_data number: %.5f  ' % (
                                        idx, label_image_count, labeled_data,
                                        unlabel_count))
                                print(
                                    'batch_idx: %d | labeled_loader number: %.3f | labeled_data number: %.5f | unlabeled_data number: %.5f  ' % (
                                        idx, label_image_count, labeled_data,
                                        unlabel_count))
                        #可以在这儿，进行阈值的更新
                        # if epoch>100:
                        #     total_count_low = sum(class_counts_low)
                        #     if total_count_low != 0:
                        #
                        #         #获取列表的中位数
                        #         median=statistics.median(class_counts_low)
                        #         # 获取最小的三个元素及其下标
                        #         smallest = heapq.nsmallest(int(num_class/3), enumerate(class_counts_low), key=lambda x: x[1])
                        #         # 分离元素和下标
                        #         values_small = [x[1] for x in smallest]
                        #         indices_small = [x[0] for x in smallest]
                        #
                        #         mean_small=sum(values_small)/len(values_small)
                        #         #如果平均数远远低于中位数 就降低阈值
                        #         if mean_small<=median/3:
                        #             for i in indices_small:
                        #                 pseudo_label_theta[i] = pseudo_label_theta[i] - 0.02
                        #                 if pseudo_label_theta[i] <= theta - 0.1:
                        #                     pseudo_label_theta[i] = theta - 0.1

                            # # 获取最大的三个元素及其下标
                            # largest = heapq.nlargest(int(num_class/3), enumerate(class_counts_low), key=lambda x: x[1])
                            # # 分离元素和下标
                            # values_large = [x[1] for x in largest]
                            # indices_large = [x[0] for x in largest]
                            # mean_large = sum(values_large) / len(indices_large)
                            # if mean_large/5 >= median :
                            #     for i in indices_large:
                            #         pseudo_label_theta[i] = pseudo_label_theta[i] + 0.01
                            #         if pseudo_label_theta[i] >= theta + 0.02:
                            #             pseudo_label_theta[i] = theta + 0.02

                label_image_count = len(labeled_loader.dataset)
                unlabel_count = len(unlabeled_loader.dataset)
                logging.info(
                    'batch_idx: %d | labeled_loader number: %.3f | labeled_data number: %.5f | unlabeled_data number: %.5f  ' % (
                        idx, label_image_count, labeled_data,
                        unlabel_count))
                print(
                    'batch_idx: %d | labeled_loader number: %.3f | labeled_data number: %.5f | unlabeled_data number: %.5f  ' % (
                        idx, label_image_count, labeled_data,
                        unlabel_count))

                if unlabel_count-current_produce_pseudo_num > 0:

                    unlabels_indices_s=[]
                    for i in range(len(unlabels_indices)):
                        if index_lists[i] == 0:
                             unlabels_indices_s.append(unlabels_indices[i])


                    unlabels_dataset = torch.utils.data.Subset(full_dataset, unlabels_indices_s)
                    unlabeled_loader = torch.utils.data.DataLoader(unlabels_dataset, batch_size=batch_size,shuffle=True)
                    print("重新构建无标签数据加载器")
                    unlabels_indices = unlabels_dataset.indices
                    # unlabel_image_count=len(unlabeled_loader.dataset)
                    # batch_size_num = int(unlabel_image_count / batch_size)
                    index_lists =[0]*len(unlabels_indices)
                    # if len(unlabels_indices) < (batch_size_num + 1) * batch_size:
                    #     num = (batch_size_num + 1) * batch_size - len(unlabels_indices)
                    #     for i in range(num):
                    #         unlabels_indices.append(0)


                    label_image_count = len(labeled_loader.dataset)
                    print("labeled_loader count:", label_image_count)

                    # label_image_count = len(labeled_data_dataset)
                    # print("labelImage count:", label_image_count)
                    unlabel_image_count = len(unlabeled_loader.dataset)
                    # unlabel_image_count =unlabel_count-labeled_data
                    print("unlabelImage count:", unlabel_image_count)
                else:
                    unlabel_image_count = 0
                    label_image_count = len(labeled_loader.dataset)
                    print("labelImage count:", label_image_count)
                    print("unlabelImage count:0")
            else:
                label_image_count = len(labeled_loader.dataset)
                print("labelImage count:", label_image_count)
                print("unlabelImage count:0")
        print("预测错误的数量：",error_num)
        print("低质量伪标签类别数量：")
        print(class_counts_low)
        total_count_current = sum(class_counts_low)
        print("当前全部可用低质量伪标签总数为：")
        print(total_count_current)
        print("伪标签阈值：")
        print(theta)
        label_image_count = len(labeled_loader.dataset)
        print("labelImage count:", label_image_count)
        unlabel_image_count = len(unlabeled_loader.dataset)
        # unlabel_image_count = unlabel_count - labeled_data
        print("unlabelImage count:", unlabel_image_count)
        # class_counts_choice = [num_samples_per_class] * num_class
        # if Semi_supervised and loss_current<0.03:
        unlabeled_loader = torch.utils.data.DataLoader(unlabels_dataset, batch_size=batch_size, shuffle=True)
        if Semi_supervised :
            # if Semi_supervised and val_acc_com_current > 80:
            # if total_count>num_samples_per_class*num_class*5 or epoch==20:
            #     Semi_supervised=False
            if epoch>75 and epoch%20==0 and is_used_all_choice_samples == False:
            # if epoch > 45 and epoch % 20 == 0 and is_used_all_choice_samples == False:
            #     pseudo_label_all_list=[]
                #用于存放本来，被选中的数量
                error_pseudo_in_high_num=0
                #加入到高质量伪标签列表中错误伪标签数据的数量
                class_counts_choice_sample = [0] * num_class
                pseudo_label_delete_index_lists = [[] for _ in range(num_class)]
                total_pseudo_label = sum(len(sublist) for sublist in pseudo_label_image_lists)
                print("伪标签列表里数据总数：", total_pseudo_label)
                if loss_current<0.03:  #说明没能打乱模型，选取的伪标签数据较好，可以加入高质量图像库
                   print("选取高质量伪标签数据，加入高质量列表中")
                   high_choice_true_label = [[] for _ in range(num_class)]
                   for i in range(len(choice_pseudo_label)):
                       if choice_pseudo_probability[i]>=pseudo_label_mean_probability_lists[choice_pseudo_label[i]]:
                           high_quality_pseudo_image_list[choice_pseudo_label[i]].append(choice_pseudo_image[i])
                           high_quality_pseudo_label_list[choice_pseudo_label[i]].append(choice_pseudo_label[i])
                           high_choice_true_label[choice_pseudo_label[i]].append(choice_true_label[i])
                           class_counts_hight[choice_pseudo_label[i]]=class_counts_hight[choice_pseudo_label[i]]+1
                           class_counts_low[choice_pseudo_label[i]] = class_counts_low[choice_pseudo_label[i]]-1
                           #1保留  0删除
                           pseudo_label_delete_index_lists[choice_pseudo_label[i]].append(1)
                           if choice_pseudo_label[i]==choice_true_label[i]:
                               true_pseudo_num=true_pseudo_num+1
                           else:
                               error_pseudo_num=error_pseudo_num+1

                           del pseudo_label_image_lists[choice_pseudo_label[i]][pseudo_label_index_list[i]]
                           del pseudo_label_label_lists[choice_pseudo_label[i]][pseudo_label_index_list[i]]
                           del pseudo_label_probability_lists[choice_pseudo_label[i]][pseudo_label_index_list[i]]
                           del pseudo_label_pre_probability_lists[choice_pseudo_label[i]][pseudo_label_index_list[i]]
                       else:
                           # class_counts_low[choice_pseudo_label[i]] = class_counts_low[choice_pseudo_label[i]] - 1
                           pseudo_label_delete_index_lists[choice_pseudo_label[i]].append(0)
                   high_total = sum(class_counts_hight)
                   if high_total==0:
                       print("高质量伪标签列表总数：0")
                   else:
                       print('高质量伪标签列表总数: %.3f | 错误伪标签数量: %.3f |比例为：%.3f' % (high_total, error_pseudo_num, error_pseudo_num / high_total))

                   # print("高质量伪标签列表中的错误伪标签数据数量为:",error_pseudo_num)

                           # pseudo_label_delete_index_lists[choice_pseudo_label[i]].append()
                           # del pseudo_label_index_lists[choice_pseudo_label[i]]
                   # high_quality_pseudo_image_list.extend(choice_pseudo_image.copy())
                   # high_quality_pseudo_label_list.extend(choice_pseudo_label.copy())
                   # for index in range(num_class):
                   #     class_counts_hight[index]=class_counts_hight[index]+class_counts_choice_samples[index]
                   #     class_counts_low[index]=class_counts_low[index]-class_counts_choice_samples[index]
                       # class_counts_choice_samples[index] = class_counts_choice_samples[index] + class_counts_choice[index]

                   # #根据下标位置从伪标签列表删除已经加入高质量列表的图像数据
                   # #保留不好的
                   # for index, label_list in enumerate(pseudo_label_index_lists):
                   #     # for i in label_list:
                   #     for i, j in enumerate(label_list):
                   #         if (pseudo_label_delete_index_lists[index][i]==1):
                   #             del pseudo_label_image_lists[index][j]
                   #             del pseudo_label_label_lists[index][j]
                   #
                   # # # #全部删除
                   # for index, label_list in enumerate(pseudo_label_index_lists):
                   #     for i in label_list:
                   #         del pseudo_label_image_lists[index][i]
                   #         del pseudo_label_label_lists[index][i]


                #计算伪标签的总数，判断是否还需要生成伪标签数据
                # 计算数据总数
                total_pseudo_label = sum(len(sublist) for sublist in pseudo_label_image_lists)
                print("伪标签列表里数据总数：",total_pseudo_label)
                choice_pseudo_image=[]
                choice_pseudo_label=[]
                choice_pseudo_probability = []
                choice_true_label = []
                pseudo_label_index_list.clear()
                for index, label_list in enumerate(pseudo_label_image_lists):
                    if is_close_pseudo_label_produce == False:
                        # if len(label_list)>=10:
                        if len(label_list)>=50:
                            random_numbers = random.sample(range(0, len(label_list)), 15)
                        else:
                            random_numbers = random.sample(range(0, len(label_list)), int(len(label_list) / 10))
                        # random_numbers = random.sample(range(0, len(label_list)),len(label_list))

                        # random_numbers = random.sample(range(0, len(label_list)), int(len(label_list) / 10))

                        # my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
                        # 使用 sort() 方法进行排序
                        random_numbers.sort()
                        # 使用 reverse() 方法进行倒序
                        random_numbers.reverse()

                        pseudo_label_index_lists[index].clear()

                        for i in random_numbers:
                            choice_pseudo_image.append(label_list[i])
                            choice_pseudo_label.append(pseudo_label_label_lists[index][i])
                            choice_true_label.append(unlabel_true_label_lists[index][i])
                            # choice_pseudo_probability.append(pseudo_label_probability_lists[index][i])
                            choice_pseudo_probability.append(pseudo_label_pre_probability_lists[index][i])
                            # pseudo_label_all_list.append(label_list[i])
                            class_counts_choice_sample[index] = class_counts_choice_sample[index] + 1
                            #把下标存放
                            pseudo_label_index_lists[index].append(i)
                            pseudo_label_index_list.append(i)
                        # else:
                        #
                        #     pseudo_label_index_lists[index].clear()
                        #     # for i in range(len(label_list)):
                        #     for i in range(len(label_list) - 1, -1, -1):
                        #         choice_pseudo_image.append(label_list[i])
                        #         choice_pseudo_label.append(pseudo_label_label_lists[index][i])
                        #         choice_true_label.append(unlabel_true_label_lists[index][i])
                        #         choice_pseudo_probability.append(pseudo_label_probability_lists[index][i])
                        #         class_counts_choice_sample[index] = class_counts_choice_sample[index] + 1
                        #         pseudo_label_index_lists[index].append(i)
                    else:
                        #如果不在产生新的伪标签数据，那么就可以使用全部数据了
                        # for i in range(len(label_list)):
                        for i in range(len(label_list) - 1, -1, -1):
                            choice_pseudo_image.append(label_list[i])
                            choice_pseudo_label.append(pseudo_label_label_lists[index][i])
                            choice_true_label.append(unlabel_true_label_lists[index][i])
                            # choice_pseudo_probability.append(pseudo_label_probability_lists[index][i])
                            choice_pseudo_probability.append(pseudo_label_pre_probability_lists[index][i])
                            class_counts_choice_sample[index] = class_counts_choice_sample[index] + 1
                            pseudo_label_index_lists[index].append(i)
                            pseudo_label_index_list.append(i)
                        is_used_all_choice_samples=True  #控制关闭 选取模块


                #删除选中的高质量列表
                # for i in range(len(high_quality_pseudo_image_choicelist)):
                # for index, label_list in enumerate(high_pseudo_label_index_lists):
                #     for i in label_list:
                #         del high_quality_pseudo_image_list[index][i]
                #         del high_quality_pseudo_label_list[index][i]
                #     class_counts_hight[index] = class_counts_hight[index]-len(label_list)
                #选取高质量的列表
                high_quality_pseudo_image_choicelist=[]
                high_quality_pseudo_label_choicelist=[]
                high_error_num=0
                for index, label_list in enumerate(high_quality_pseudo_image_list):
                    if len(label_list) >= 50:
                        random_numbers = random.sample(range(0, len(label_list)), 15)
                    else:
                        random_numbers = random.sample(range(0, len(label_list)), int(len(label_list) / 3))
                    # my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
                    # 使用 sort() 方法进行排序
                    random_numbers.sort()
                    # 使用 reverse() 方法进行倒序
                    random_numbers.reverse()

                    # pseudo_label_index_lists[index].clear()
                    class_counts_hight_choice[index]=0
                    high_pseudo_label_index_lists[index].clear()
                    for i in random_numbers:
                        high_quality_pseudo_image_choicelist.append(label_list[i])
                        high_quality_pseudo_label_choicelist.append(high_quality_pseudo_label_list[index][i])
                        class_counts_hight_choice[index]=class_counts_hight_choice[index]+1
                        high_pseudo_label_index_lists[index].append(i)
                            # if high_choice_true_label[index][i]!=high_quality_pseudo_label_list[index][i]:
                            #     high_error_num=high_error_num+1
                            # choice_true_label.append(unlabel_true_label_lists[index][i])
                            # choice_pseudo_probability.append(pseudo_label_probability_lists[index][i])
                            # pseudo_label_all_list.append(label_list[i])
                            # class_counts_choice_sample[index] = class_counts_choice_sample[index] + 1
                            # 把下标存放
                            # pseudo_label_index_lists[index].append(i)

                #计算全体的概率平均值，用于二次筛选标签
                print("从高质量列表中选到错误数据数量：",high_error_num)


                #数据保存起来，用于下一次计算高质量伪标签的数量
                class_counts_choice_samples=class_counts_choice_sample
                print("随机更新无标签图片")
                if total_pseudo_label < close_condition / 2:
                    print("伪标签数量减少，开启伪标签数据生成")
                    is_close_pseudo_label_produce = False
                for i in range(num_class):
                    # class_counts_choice[i] = class_counts_origin[i]+class_counts_choice_samples[i]
                    class_counts_choice[i] = class_counts_origin[i] + class_counts_hight_choice[i] + class_counts_choice_samples[i]
                total_count = sum(class_counts_choice)
                # del labeled_dataset
                # del labeled_loader

                # #防止批次出现单个或者少个的清空，情绪批次补全
                # completion_image_list=[]
                # completion_label_list=[]
                # #进行批次补全
                # completion_num = (batch_size - ((total_count-num_samples_per_class*num_class) % batch_size))
                # # completion_num = (batch_size - ((total_count) % batch_size))
                # completion_random_numbers = random.sample(range(0, len(labeled_data_origin)), completion_num)
                # print("填补批次空挡数量：",completion_num)
                # for i in completion_random_numbers:
                #     completion_image_list.append(labeled_data_origin[i])
                #     completion_label_list.append(labeled_data_origin_label[i])
                #
                # # #重新构建标签数据加载器
                # pseudo_dataset = torch.utils.data.TensorDataset(torch.stack(choice_pseudo_image+high_quality_pseudo_image_choicelist+completion_image_list),
                #                                               torch.stack(choice_pseudo_label+high_quality_pseudo_label_choicelist+completion_label_list))
                # # 创建新的数据加载器
                # pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=batch_size,shuffle=True)


                pseudo_dataset_len = len(choice_pseudo_image)+len(high_quality_pseudo_image_choicelist)
                labeled_data_len=len(labeled_data_origin)
                completion_image_label_list = []
                completion_label_label_list = []
                # 进行批次补全
                # completion_num = ()
                #如果有标签比较长  ，伪标签比较长
                count_total=0
                if pseudo_dataset_len!=labeled_data_len:
                    if pseudo_dataset_len>labeled_data_len:

                        gapnum = pseudo_dataset_len - labeled_data_len
                        if gapnum<labeled_data_len:
                            pseudo_data_completion_random_numbers = random.sample(range(0, labeled_data_len), gapnum)
                            pseudo_data_completion_random_numbers = pseudo_data_completion_random_numbers * (
                                        gapnum // labeled_data_len) + pseudo_data_completion_random_numbers[
                                                                      :gapnum % labeled_data_len]
                        else:
                            pseudo_data_completion_random_numbers=list(range(labeled_data_len))
                            pseudo_data_completion_random_numbers = pseudo_data_completion_random_numbers * (
                                    gapnum // labeled_data_len) + pseudo_data_completion_random_numbers[
                                                                  :gapnum % labeled_data_len]
                        print("填补批次空挡数量：", pseudo_data_completion_random_numbers)
                        for i in pseudo_data_completion_random_numbers:
                            completion_image_label_list.append(labeled_data_origin[i])
                            completion_label_label_list.append(labeled_data_origin_label[i])
                        count_total=pseudo_dataset_len
                        # indices = list(range(labeled_data_len))
                        # if len(indices) < pseudo_dataset_len:
                        #     indices = indices * (pseudo_dataset_len // len(indices)) + indices[:pseudo_dataset_len % len(indices)]

                        # labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True,
                        #                            sampler=SubsetRandomSampler(indices))
                    else:
                        gapnum = labeled_data_len - pseudo_dataset_len
                        if gapnum < labeled_data_len:
                            pseudo_data_completion_random_numbers = random.sample(range(0, labeled_data_len), gapnum)
                            pseudo_data_completion_random_numbers = pseudo_data_completion_random_numbers * (gapnum // pseudo_dataset_len) + pseudo_data_completion_random_numbers[:gapnum % pseudo_dataset_len]
                        else:
                            pseudo_data_completion_random_numbers=list(range(labeled_data_len))
                            pseudo_data_completion_random_numbers = pseudo_data_completion_random_numbers * (
                                        gapnum // pseudo_dataset_len) + pseudo_data_completion_random_numbers[
                                                                        :gapnum % pseudo_dataset_len]
                        print("填补批次空挡数量：", pseudo_data_completion_random_numbers)
                        for i in pseudo_data_completion_random_numbers:
                            completion_image_label_list.append(labeled_data_origin[i])
                            completion_label_label_list.append(labeled_data_origin_label[i])
                        count_total = labeled_data_len
                else:
                    count_total = labeled_data_len
                            # #重新构建标签数据加载器
                        # pseudo_dataset = torch.utils.data.TensorDataset(torch.stack(choice_pseudo_image + high_quality_pseudo_image_choicelist + completion_image_list),
                        #                                                 torch.stack(choice_pseudo_label + high_quality_pseudo_label_choicelist + completion_label_list))
                        # # 创建新的数据加载器
                        # pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=batch_size,
                        #                                             shuffle=True)

                #防止批次出现单个或者少个的清空，情绪批次补全
                completion_image_list=[]
                completion_label_list=[]
                #进行批次补全
                completion_num = (batch_size - ((count_total-num_samples_per_class*num_class) % batch_size))
                # completion_num = (batch_size - ((total_count) % batch_size))
                completion_random_numbers = random.sample(range(0, len(labeled_data_origin)), completion_num)
                print("填补批次空挡数量：",completion_num)
                for i in completion_random_numbers:
                    completion_image_list.append(labeled_data_origin[i])
                    completion_label_list.append(labeled_data_origin_label[i])


                if pseudo_dataset_len>labeled_data_len:
                    labeled_dataset = torch.utils.data.TensorDataset(
                        torch.stack(labeled_data_origin+completion_image_label_list + completion_image_list),
                        torch.stack(labeled_data_origin_label+completion_label_label_list+ completion_label_list))
                    # 创建新的数据加载器
                    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
                    pseudo_dataset = torch.utils.data.TensorDataset(torch.stack(choice_pseudo_image + high_quality_pseudo_image_choicelist + completion_image_list),
                        torch.stack(choice_pseudo_label + high_quality_pseudo_label_choicelist + completion_label_list))
                    # 创建新的数据加载器
                    pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)
                if pseudo_dataset_len<labeled_data_len:
                    labeled_dataset = torch.utils.data.TensorDataset(torch.stack(labeled_data_origin +completion_image_list),
                        torch.stack(labeled_data_origin_label  + completion_label_list))
                    # 创建新的数据加载器
                    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
                    pseudo_dataset = torch.utils.data.TensorDataset(torch.stack( choice_pseudo_image + high_quality_pseudo_image_choicelist +completion_image_label_list+ completion_image_list),
                        torch.stack(choice_pseudo_label + high_quality_pseudo_label_choicelist +completion_label_label_list+ completion_label_list))
                    # 创建新的数据加载器
                    pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)

                if pseudo_dataset_len==labeled_data_len:
                    labeled_dataset = torch.utils.data.TensorDataset(torch.stack(labeled_data_origin +completion_image_list),
                        torch.stack(labeled_data_origin_label  + completion_label_list))
                    # 创建新的数据加载器
                    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
                    pseudo_dataset = torch.utils.data.TensorDataset(torch.stack( choice_pseudo_image + high_quality_pseudo_image_choicelist + completion_image_list),
                        torch.stack(choice_pseudo_label + high_quality_pseudo_label_choicelist + completion_label_list))
                    # 创建新的数据加载器
                    pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)
                # # 如果pseudo_dataset的长度不足batch_size，会自动重复选取以填充到满足batch_size
                # pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=False,
                #                            sampler=SubsetRandomSampler(range(len(pseudo_dataset))))
                #
                # # 或者使用RandomSampler，它会随机选择样本，可能会出现重复
                # pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=False,
                #                            sampler=RandomSampler(pseudo_dataset, replacement=True, num_samples=batch_size))
                #
                # indices = list(range(len(pseudo_dataset)))
                # if len(indices) < 400:
                #     indices = indices * (400 // len(indices)) + indices[:400 % len(indices)]
                # else:
                #     indices = indices[:400]
                # pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=False,
                #                            sampler=SubsetRandomSampler(indices))


                # 重新构建标签数据加载器
                # labeled_dataset = torch.utils.data.TensorDataset(
                #     torch.stack(labeled_data_origin+choice_pseudo_image + high_quality_pseudo_image_choicelist + completion_image_list),
                #     torch.stack(labeled_data_origin_label+choice_pseudo_label + high_quality_pseudo_label_choicelist + completion_label_list))
                # # 创建新的数据加载器
                # labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
        # print("随机更新后的类别数量(每个类别随机选择固定数量的图片)：")
        # print(class_counts_choice)
        # total_count = sum(class_counts_choice)
        # print("训练总数：")
        # print(total_count)
        print("原始数据数量列表：")
        print(class_counts_origin)
        print("高质量伪标签数据数量列表：")
        print(class_counts_hight)
        print("被选中的高质量伪标签列表总数")
        print(class_counts_hight_choice)
        print("当前选中的低质量伪标签数据数量列表：")
        print(class_counts_choice_samples)
        print("总共低质量伪标签数据数量列表：")
        print(class_counts_low)
        print("每个类别的阈值：")
        print(pseudo_label_theta)
        for i in range(num_class):
            # class_counts_choice[i] = class_counts_origin[i] + class_counts_choice_samples[i]
            class_counts_choice[i] = class_counts_origin[i] + class_counts_hight_choice[i] + class_counts_choice_samples[i]
        print("当前训练数据数量列表(原始数量+当前抽取的高质量伪标签数量+当前选择的低质量伪标签数量)：")
        print(class_counts_choice)
        #更新损失权重
        class_counts_weight=class_counts_choice
        weights = torch.Tensor([1 / count for count in class_counts_weight])
        print("当前的全部训练样本数量")
        total_count = sum(class_counts_choice)
        print(total_count)
        if epoch == 75:
            for index, label_list in enumerate(pseudo_label_probability_lists):
                # 计算平均值或者方差
                if len(label_list) != 0:
                    pseudo_label_mean_probability_lists[index] = sum(label_list) / len(label_list)
                else:
                    pseudo_label_mean_probability_lists[index] = theta
        print("计算每个类别的平均值：")
        print(pseudo_label_mean_probability_lists)

        augmentations = [
            # transforms.GaussianBlur(kernel_size=3),
            transforms.RandomResizedCrop(310, scale=(0.8, 1.0)),
            transforms.RandomErasing(p=0.5, scale=(0.1, 0.2), ratio=(0.5, 3.3), value='random'),
            # transforms.Grayscale(num_output_channels=3),  # 灰度图像变换
            # transforms.RandomAutocontrast(p=0.5),  # 图像的像素值分布均匀化
            transforms.RandomInvert(p=0.5),  # 随机反色处理
            # transforms.RandomPosterize(5, p=0.5),
            transforms.RandomAdjustSharpness(1.5, p=0.5),
            transforms.RandomSolarize(threshold=192.0),  # 随机像素值取反
        ]
        # 随机选取三种不重复的增强方法
        chosen_augmentations_all = random.sample(augmentations, 2)
        chosen_augmentations_strong = random.sample(augmentations, 4)
        transform_train = transforms.Compose([
            transforms.Resize((310, 310)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-30, 30)),
            # *chosen_augmentations_strong,
            # transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_train_all = transforms.Compose([
            transforms.Resize((310, 310)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(310, scale=(0.8, 1.0)),
            transforms.RandomErasing(p=0.5, scale=(0.1, 0.2), ratio=(0.5, 3.3), value='random'),
            transforms.RandomRotation(degrees=(-30, 30)),
            # *chosen_augmentations_all,
            # *chosen_augmentations_strong,
            # transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])




        if ( epoch > 1 and epoch % 10 == 0):
            print("更新数据增强方式")
            labeled_loader.transform = transform_train_all  # 在未生成伪标签之前，尽可能的提高模型泛化性和准确性
            pseudo_loader.transform = transform_train

        if (epoch >80 and Semi_supervised==True  and epoch % 10 == 0):
            labeled_loader.transform = transform_train_all
            pseudo_loader.transform = transform_train
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = 0.03

    # 获取当前时间的时间戳
    current_timestamp = time.time()

    # 将时间戳转换为结构化时间
    current_struct_time = time.localtime(current_timestamp)

    # 输出当前时间
    print("训练结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", current_struct_time))
    # 恢复标准输出到默认
    sys.stdout = sys.__stdout__
if __name__ == '__main__':

    # data_path = 'G://论文//datasets//datasets//CUB100'
    # data_path = 'G://论文//datasets//datasets//NWPU-RESISC45'
    # data_path = 'G://论文//datasets//datasets//UCMerced_LandUse//Images'
    # data_path = 'G://论文//datasets//datasets//AID//AID_dataset//AID'
    # data_path = 'G://论文//datasets//datasets//RSI-CB256//RSI-CB256//RSI-CB256'

    # data_path ='G://论文//datasets//datasets//CUB_200_2011//CUB_200_2011//CUB2002011'
    # data_path = '/home/fengjf/data/lhx/code/dataset/CUB100'
    # data_path = '/home/fengjf/data/lhx/code/dataset/NWPU-RESISC45'
    # data_path = '/home/fengjf/data/lhx/code/dataset/UCMerced_LandUse'
    # data_path = '/home/fengjf/data/lhx/code/dataset/12class'
    # data_path = '/home/fengjf/data/lhx/code/dataset/AID'
    #data_path = '/home/fengjf/data/lhx/code/dataset/RSI-CB256'
    data_path = '/home/fengjf/data/lhx/code/dataset/newCLRS'

    # data_path ='/media/fengjf/13F0915782D333AF/lhx/dataset/CUB100'
    # data_path ='/media/fengjf/13F0915782D333AF/lhx/dataset/NWPU-RESISC45'
    # data_path ='/media/fengjf/13F0915782D333AF/lhx/dataset/UCMerced_LandUse'
    # data_path ='/media/fengjf/13F0915782D333AF/lhx/dataset/AID'
    # data_path='/home/cqupt/lhx/dataset/CUB100'
    # data_path = 'G://论文//datasets//datasets//CUB100//FGVC_Aircraft'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train(nb_epoch=201,             # number of epoch
             batch_size=15,         # batch size
             # store_name='CUB100_Semi',     # folder for output
             store_name='251test',
             num_class=25,
             Semi_supervised=True,
             smalldataset=False,
             isshowimage=False,
             num_samples_per_class=1,
             train_image_total=105,  #21  5 252  1   30 800     45 1350
             resume=False,          # resume training from checkpoint
             start_epoch=1,         # the start epoch number when you resume the training
             model_path='',
             data_path = data_path)         # the saved model where you want to resume the training
    torch.cuda.empty_cache()

