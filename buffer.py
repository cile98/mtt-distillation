import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug, get_racing_dataset
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    if args.dataset != 'CarRacing':
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    else:
        channel, im_size, num_classes, class_names, dst_train, loader_train_dict, class_map, class_map_inv = get_racing_dataset(args.data_path, args=args)
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    dst_train = TensorDataset(images_all.detach(), labels_all.detach())
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' set augmentation for whole-dataset training '''
    #args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    #args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    #print('DC augmentation parameters: \n', args.dc_aug_param)

    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr)  # was SGD
        scheduler = torch.optim.lr_scheduler.ExponentialLR(teacher_optim, gamma=0.1)
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=False)

            #test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
            #                            criterion=criterion, args=args, aug=False)

            print("Itr: {}\tEpoch: {}\tTrain Loss {}\t".format(it, e, train_loss)) # test_acc

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
            if (e + 1) % 100 == 0:
                scheduler.step()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
            torch.save(teacher_net, os.path.join(save_dir, "net_{}.t7".format(n)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CarRacing', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='CarRacing', help='model')
    parser.add_argument('--num_experts', type=int, default=1, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=512, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=512, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='False', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=250)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=1)

    args = parser.parse_args()
    main(args)


