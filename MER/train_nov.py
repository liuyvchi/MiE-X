import argparse
import time
import torch
import dlib
import numpy as np

import sys;
# sys.path.append("../Model/")
# sys.path.append("../Utils/")

from Model.Model_GAN import resnet18, GAN_npic
from Utils.Dataloader_GAN import GAN_loader, New_loader, Micro_loader
from PIL import Image
from torch import stack, cuda, nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Function
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from datetime import datetime, timedelta

random.seed(1)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


class ReverseLayerF(torch.autograd.Function):
    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, gradOutput):
        self.coeff = np.float(
            2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high - self.low) + self.low)
        return -self.coeff * gradOutput


ReverseLayerF = ReverseLayerF()


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return "{:0>2} hrs, {:0>2} mins, {:0>2} secs".format(hours, minutes, seconds)


def output_iteration(loss, i, time, totalitems):
    avgBatchTime = time / (i + 1)
    estTime = avgBatchTime * (totalitems - i)

    print("Iteration: {:0>8},Elapsed Time: {},Estimated Time Remaining: {},Loss:{}".format(i, timedelta_string(time),
                                                                                           timedelta_string(estTime),
                                                                                           loss))


def train(train_dataset, train_dataloader, model, criterion, optimizer, epoch, print_freq=50):
    for i in range(2):
        model['resnet'][i].train()
    model['classifier'].train()
    model['discriminator'].train()
    correct = 0
    total_correct = 0
    total_samples = 0
    FP, FN, TP, TN = 0, 0, 0, 0

    startTime = datetime.now()

    for i, sample in enumerate(train_dataloader):
        input, label, domain_label = sample['images'], sample['label'], sample['domain_label']
        input, label, domain_label = input.cuda(), label.cuda(), domain_label.cuda()
        batch_size = input.size()[0]
        output_encoder = []
        for k in range(2):
            output_encoder.append(model['resnet'][k](input[:, k]))

        features = torch.cat(output_encoder, 1)

        reverse_feature = ReverseLayerF(features)

        domain_output = model['discriminator'](reverse_feature).squeeze(dim=1)
        classification_output = model['classifier'](features)

        domain_loss = criterion['domain'](domain_output, domain_label.float())
        classification_loss = criterion['classification'](classification_output, label)
        loss = classification_loss

        _, preds = torch.max(classification_output, dim=1)

        # print(preds, label)

        correct = float((label.int() == preds.int()).sum())
        total_correct += correct
        total_samples += batch_size
        accuracy = correct / batch_size

        optimizer.zero_grad()
        loss.backward()

        Clip_Norm = 1
        nn.utils.clip_grad_norm_(model['resnet'][0].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(model['resnet'][1].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(model['classifier'].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(model['discriminator'].parameters(), Clip_Norm, norm_type=2)

        optimizer.step()

        matrix = confusion_matrix(label, preds, labels=[0, 1, 2])
        FP += matrix.sum(axis=0) - np.diag(matrix)
        FN += matrix.sum(axis=1) - np.diag(matrix)
        TP += np.diag(matrix)
        TN += matrix.sum() - (FP + FN + TP)

        sampleNumber = i * 32

        if i % print_freq == 0:
            print('Train:\t'
                  'Epoch:[{0}][{1}/{2}]   \t'
                  'Acc: {acc:.3f}\t'
                  'Label_Loss: {c_loss:.4f}\t'
                  'Domain_Loss: {d_loss:.4f}\t'
                  'Loss: {loss:.4f}\t'.format(
                epoch, i + 1, len(train_dataloader), acc=accuracy, c_loss=classification_loss, d_loss=domain_loss,
                loss=loss))
            currentTime = datetime.now()
            output_iteration(loss.cpu().detach().numpy(), sampleNumber, currentTime - startTime,
                             len(train_dataset))

    total_accuracy = total_correct / total_samples
    print("epoch training accuracy:", total_accuracy)

    f1_s = np.ones([3])
    deno = (2 * TP + FP + FN)
    for f in range(3):
        if deno[f] != 0:
            f1_s[f] = float((2 * TP[f]) / (2 * TP[f] + FP[f] + FN[f]))
        else:
            f1_s[f] = 1


def validate(validate_dataloader, model, criterion):
    for i in range(2):
        model['resnet'][i].eval()
    model['classifier'].eval()
    model['discriminator'].eval()

    correct = 0
    total_samples = 0

    total_loss = 0
    total_loss_c = 0
    total_loss_d = 0
    FP, FN, TP, TN = 0, 0, 0, 0

    with torch.no_grad():
        for i, sample in enumerate(validate_dataloader):
            input, label, domain_label = sample['images'], sample['label'], sample['domain_label']
            input, label, domain_label = input.cuda(), label.cuda(), domain_label.cuda()
            batch_size = input.size()[0]
            output_encoder = []
            for k in range(2):
                output_encoder.append(model['resnet'][k](input[:, k]))

            features = torch.cat(output_encoder, 1)

            reverse_feature = ReverseLayerF(features)

            domain_output = model['discriminator'](reverse_feature).squeeze(dim=1)
            classification_output = model['classifier'](features)

            domain_loss = criterion['domain'](domain_output, domain_label.float())
            classification_loss = criterion['classification'](classification_output, label)
            loss = domain_loss * 1 + classification_loss

            total_loss_c += classification_loss
            total_loss_d += domain_loss
            total_loss += loss

            _, preds = torch.max(classification_output, dim=1)

            correct += float((label.int() == preds.int()).sum())
            total_samples += batch_size

            matrix = confusion_matrix(label, preds, labels=[0, 1, 2])
            FP += matrix.sum(axis=0) - np.diag(matrix)
            FN += matrix.sum(axis=1) - np.diag(matrix)
            TP += np.diag(matrix)
            TN += matrix.sum() - (FP + FN + TP)

    accuracy = correct / total_samples
    total_loss = total_loss / total_samples
    f1_s = np.ones([3])
    deno = (2 * TP + FP + FN)
    for f in range(3):
        if deno[f] != 0:
            f1_s[f] = (2 * TP[f]) / (2 * TP[f] + FP[f] + FN[f])
        else:
            f1_s[f] = 1

    f1_score = np.mean(f1_s)
    recall = np.mean(TP / (TP + FN))

    print('Validation:\t'
          'Acc: {acc:.3f}\t'
          'Label_Loss: {c_loss:.4f}\t'
          'Domain_Loss: {d_loss:.4f}\t'
          'Loss: {loss:.4f}\t'
          'Recall: {recall:.4f}\t'
          'f1_score: {f1_score:.4f}\t'.format(acc=accuracy, c_loss=total_loss_c, d_loss=total_loss_d, loss=total_loss,
                                              recall=recall, f1_score=f1_score))

    return f1_score, recall, accuracy


def test(test_dataloader, model, criterion):
    for i in range(2):
        model['resnet'][i].eval()
    model['classifier'].eval()
    model['discriminator'].eval()

    correct = 0
    total_samples = 0

    total_loss = 0
    total_loss_c = 0
    total_loss_d = 0
    FP, FN, TP, TN = 0, 0, 0, 0

    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            input, label, domain_label = sample['images'], sample['label'], sample['domain_label']
            input, label, domain_label = input.cuda(), label.cuda(), domain_label.cuda()
            batch_size = input.size()[0]
            output_encoder = []
            for k in range(2):
                output_encoder.append(model['resnet'][k](input[:, k]))

            features = torch.cat(output_encoder, 1)

            reverse_feature = ReverseLayerF(features)

            domain_output = model['discriminator'](reverse_feature).squeeze(dim=1)
            classification_output = model['classifier'](features)

            domain_loss = criterion['domain'](domain_output, domain_label.float())
            classification_loss = criterion['classification'](classification_output, label)
            loss = domain_loss * 1 + classification_loss

            total_loss_c += classification_loss
            total_loss_d += domain_loss
            total_loss += loss

            _, preds = torch.max(classification_output, dim=1)
            print(preds, label)

            correct += float((label.int() == preds.int()).sum())
            total_samples += batch_size

            matrix = confusion_matrix(label, preds, labels=[0, 1, 2])
            FP += matrix.sum(axis=0) - np.diag(matrix)
            FN += matrix.sum(axis=1) - np.diag(matrix)
            TP += np.diag(matrix)
            TN += matrix.sum() - (FP + FN + TP)

    accuracy = correct / total_samples
    total_loss = total_loss / total_samples
    f1_s = np.ones([3])
    deno = (2 * TP + FP + FN)
    for f in range(3):
        if deno[f] != 0:
            f1_s[f] = (2 * TP[f]) / (2 * TP[f] + FP[f] + FN[f])
        else:
            f1_s[f] = 1

    f1_score = np.mean(f1_s)
    recall = np.mean(TP / (TP + FN))

    print('test:\t'
          'Acc: {acc:.3f}\t'
          'Label_Loss: {c_loss:.4f}\t'
          'Domain_Loss: {d_loss:.4f}\t'
          'Loss: {loss:.4f}\t'
          'Recall: {recall:.4f}\t'
          'f1_score: {f1_score:.4f}\t'.format(acc=accuracy, c_loss=total_loss_c, d_loss=total_loss_d, loss=total_loss,
                                              recall=recall, f1_score=f1_score))

    return f1_score, recall, accuracy


def build_model_2pic(num_classes=3):
    model = GAN_npic(fc=[512 * 2, 128, 32, 3])

    model_resnet = resnet18(pretrained=True)
    pretrained_dict = model_resnet.state_dict()
    model_dict = model['resnet'][0].state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    for i in range(2):
        model['resnet'][i].load_state_dict(model_dict)

    return model


def set_gpus(args):
    # get gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    # set gpu ids
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2,')
    parser.add_argument('--DATA_ROOT', type=str, default='/home/user/Yuchi/iccv2019/GANimation-master/',
                        help='path to generated faces')
    parser.add_argument('--train_data_path', type=str, help='path to training data')
    # parser.add_argument('--train_label_path', type=str, help='path to training labels')
    parser.add_argument('--validation_data_path', type=str, help='path to vali data')
    # parser.add_argument('--validation_label_path', type=str, help='path to vali labels')
    parser.add_argument('--test_data_path', type=str, help='path to test data')
    parser.add_argument('--test_label_path', type=str, help='path to test labels')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--nepochs_no_decay', type=int, default=20, help='# of epochs at starting learning rate')
    parser.add_argument('--nepochs_decay', type=int, default=10,
                        help='# of epochs to linearly decay learning rate to zero')
    parser.add_argument('--lr_E', type=float, default=1e-5, help='initial learning rate for Encoder')
    parser.add_argument('--lr_C', type=float, default=1e-5, help='initial learning rate for Classifer')
    parser.add_argument('--lr_D', type=float, default=1e-5, help='initial learning rate for Discriminator')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    args = parser.parse_args()

    print(args)

    set_gpus(args)

    start_time = time.time()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 256, 256
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.ColorJitter(hue=0.2),
        transforms.RandomCrop((224, 224), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dataset = New_loader(args, args.train_data_path, sample_num= 14997, mix_type=1, transform=transform,
                               Micro_set=False)
    test_dataset = Micro_loader(args, args.test_data_path, args.test_label_path, transform=transform, Micro_set=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.num_workers)

    model = build_model_2pic()

    model['discriminator'] = nn.Sequential(
        nn.Linear(1024, 128),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(128, 32),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(32, 1)
    )

    for i in range(2):
        model['resnet'][i].cuda()
    model['classifier'].cuda()
    model['discriminator'].cuda()

    model['classifier'] = model['classifier'].apply(weight_init)
    model['discriminator'] = model['discriminator'].apply(weight_init)

    criterion = {}
    criterion['classification'] = torch.nn.CrossEntropyLoss()
    criterion['domain'] = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': model['resnet'][0].parameters(), 'lr': args.lr_E},
        {'params': model['resnet'][1].parameters(), 'lr': args.lr_E},
        {'params': model['classifier'].parameters(), 'lr': args.lr_C},
        {'params': model['discriminator'].parameters(), 'lr': args.lr_D},
    ], weight_decay=args.weight_decay)

    best_vali_f1s = 0
    choosen_test_f1 = 0
    choosen_test_acc = 0
    choosen_test_recall = 0

    for epoch in range(args.nepochs_no_decay + args.nepochs_decay):
        print('epoch:', epoch, "################################################################")
        if epoch >= args.nepochs_no_decay:
            lr_decay = 1 - (epoch - args.nepochs_no_decay) / args.nepochs_decay
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = args.lr_E * lr_decay
                print(optimizer.param_groups[i]['lr'])
        start = time.time()
        train(train_dataset, train_loader, model, criterion, optimizer, epoch)
        test_f1_score, test_recall, test_acc = test(test_loader, model, criterion)

        print("cost for the last epoch:", time.time() - start)
        if test_f1_score >= best_vali_f1s:
            best_vali_f1s = test_f1_score
            choosen_test_f1 = test_f1_score
            choosen_test_acc = test_acc
            choosen_test_recall = test_recall
        print ('best_f1_score_vali:', best_vali_f1s, 'choosen_test_f1_score:', choosen_test_f1, 'choosen_test_recall:',
               choosen_test_recall, 'choosen_test_acc:', choosen_test_acc)

        # test(validation_loader, model, criterion)


if __name__ == '__main__':
    main()