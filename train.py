'''
进行训练
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import json
from math import ceil
import argparse
import copy
from ImageDataset import ImageDataset
from SimpleNet import SimpleNet
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='log')
def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes, use_gpu):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  #导入权重
    best_acc = 0.0
    device = torch.device('cuda' if use_gpu else 'cpu')
    for epoch in range(args.start_epoch, num_epochs):
        # 每一个epoch中都有一个训练和一个验证过程(Each epoch has a training and validation phase)
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step(epoch)
                # 设置为训练模式(Set model to training mode)
                model.train()
            else:
                # 设置为验证模式(Set model to evaluate mode)
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            tic_batch = time.time()
            # 在多个batch上依次处理数据(Iterate over data)
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 梯度置零(zero the parameter gradients)
                optimizer.zero_grad()
                # 前向传播(forward)
                # 训练模式下才记录梯度以进行反向传播(track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # 训练模式下进行反向传播与梯度下降(backward + optimize only if in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率(statistics)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / (i * args.batch_size + inputs.size(0))
                batch_acc = running_corrects.double() / (i * args.batch_size + inputs.size(0))

                if phase == 'train' and (i + 1) % args.print_freq == 0:
                    print(
                        '[Epoch {}/{}]-[batch:{}/{}] lr:{:.6f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f} sec/batch'.format(
                            epoch + 1, num_epochs, i + 1, ceil(dataset_sizes[phase] / args.batch_size),
                            scheduler.get_lr()[0], phase, batch_loss, batch_acc,
                            (time.time() - tic_batch) / args.print_freq))
                    tic_batch = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if epoch == 0:
                os.remove('result.txt')
            with open('result.txt', 'a') as f:
                f.write('Epoch:{}/{} {} Loss: {:.4f} Acc: {:.4f} \n'.format(epoch + 1, num_epochs, phase, epoch_loss,
                                                                            epoch_acc))

            print('{}, Epoch-{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch, epoch_loss, epoch_acc))

            writer.add_scalar(phase + '/Loss', epoch_loss, epoch)
            writer.add_scalar(phase + '/Acc', epoch_acc, epoch)

        if (epoch + 1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))

        # 深拷贝模型(deep copy the model)
        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # 将model保存为graph
    writer.add_graph(model, (inputs,))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Accuracy: {:4f}'.format(best_acc))

    # 载入最佳模型参数(load best model weights)
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='classification')
    # 图片数据的根目录(Root catalog of images)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='output')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    args = parser.parse_args()

    # read data
    dataloders, dataset_sizes, class_names = ImageDataset(args)

    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    model = SimpleNet()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            model.load_state_dict(torch.load(args.resume))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = torch.nn.DataParallel(model)
        model.to(torch.device('cuda'))
    else:
        model.to(torch.device('cpu'))

    # 用交叉熵损失函数(define loss function)
    criterion = nn.CrossEntropyLoss()

    # 梯度下降(Observe that all parameters are being optimized)
    optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00004)

    # Decay LR by a factor of 0.98 every 1 epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

    model = train_model(args=args,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.num_epochs,
                        dataset_sizes=dataset_sizes,
                        use_gpu=use_gpu)

    torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model_wts.pth'))

    writer.close()

