import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import vis_caps
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model_cap_final import Protopool_cap
from utils import mixup_data, find_high_activation_crop
import os
import matplotlib.pyplot as plt
import cv2

from utils import mixup_data, compute_proto_layer_rf_info_v2, compute_rf_prototype

def save_model(model, path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }, path)

def make_correct_class(idx, proto_correct_class):
    result = proto_correct_class.clone()
    for i in range(idx.shape[0]):
        idxs = idx[i,:]
        result[i,idxs] += 1
    return result.cuda()
    
def load_model(model, path, device):
    if device.type == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'\033[0;32mLoad model form: {path}\033[0m')
    return model, checkpoint['epoch']


def adjust_learning_rate(optimizer, rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def learn_model(opt: Optional[List[str]]) -> None:
    parser = argparse.ArgumentParser(description='PrototypeGraph')
    parser.add_argument('--data_train', help='Path to train data')
    parser.add_argument('--data_push', help='Path to push data')
    parser.add_argument('--data_test', help='Path to tets data')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='input batch size for training (default: 80)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--push_start', type=int, default=20)
    parser.add_argument('--when_push',default = False)
    parser.add_argument('--no_cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--num_descriptive', type=int, default=10)
    parser.add_argument('--num_prototypes', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=200)

    parser.add_argument('--arch', type=str, default='resnet34')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')
    parser.add_argument('--prototype_activation_function',
                        type=str, default='log')

    parser.add_argument('--use_thresh', action='store_true')
    parser.add_argument('--earlyStopping', type=int, default=None,
                        help='Number of epochs to early stopping')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--results', default='./results',
                        help='Path to dictionary where will be save results.')
    parser.add_argument('--ppnet_path', default=None)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup_time', default=100, type=int)
    parser.add_argument('--gumbel_time', default=10, type=int)
    parser.add_argument('--proto_depth', default=128, type=int)
    parser.add_argument('--last_layer', action='store_true')
    parser.add_argument('--inat', action='store_true')
    parser.add_argument('--mixup_data', action='store_true')
    parser.add_argument('--push_only', action='store_true')
    parser.add_argument('--gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    parser.add_argument('--proto_img_dir', type=str, default='img')
    parser.add_argument('--pp_ortho', action='store_true')
    parser.add_argument('--pp_gumbel', action='store_true')
    parser.add_argument('--drop_last',default = True) #doubt but keep it true as in protopool
    parser.add_argument('--type2', default = False, type = bool)
    parser.add_argument('--cap_start',default = None, type = int)
    parser.add_argument('--fine_tune_epoch',default = 15,type = int)
    parser.add_argument('--capl',default = 0, type = float)
    parser.add_argument('--capcoef',default = 0, type = float)
    parser.add_argument('--cap_all',default = False, type = bool)
    parser.add_argument('--only_warmuptr',default = False, type = bool)
    parser.add_argument('--topk_loss',default = False, type = bool)
    parser.add_argument('--sep',default = False, type = bool)
    parser.add_argument('--k_top',default = 10, type = int)
    
    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    top_k = args.k_top
    cap_width = args.capl # input in protopool_cap
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')

    start_val = 1.3
    end_val = 10 ** 3
    epoch_interval = args.gumbel_time
    alpha = (end_val / start_val) ** 2 / epoch_interval

    def lambda1(epoch): 
        return start_val * np.sqrt(alpha *(epoch)) if epoch < epoch_interval else end_val

    clst_weight = 0.8
    sep_weight = -0.08
    #tau = 1 tau is set as 0.5 in model

    if args.seed is None:  # 1234
        args.seed = np.random.randint(10, 10000, size=1)[0]
        #args.seed = 3407
    torch.manual_seed(args.seed)
    print('current seed is:', args.seed)
    kwargs = {}
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        kwargs.update({'num_workers': 2, 'pin_memory': True})

    transforms_train_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transforms_analy = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(
        args.data_train,
        transforms_train_test,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last,
        **kwargs)

    test_dataset = datasets.ImageFolder(
        args.data_test,
        transforms_train_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        **kwargs)
    
    # added caps with 
    caps_anal_dataset = datasets.ImageFolder(
        args.data_push,transforms_analy)

    caps_anal_loader = torch.utils.data.DataLoader(
        caps_anal_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=False)
    print('Architecture:', args.arch)
    print('top-k cluster loss with k:',top_k)
    print('train cap with new arch')
    print('cap_width:', cap_width)
    print('cap_coef:',args.capcoef)
    print('cap_all:', args.cap_all)
    print('only_warmup_tr',args.only_warmuptr)
    
    ###########################
    model = Protopool_cap(
        num_prototypes=args.num_prototypes,
        num_descriptive=args.num_descriptive,
        num_classes=args.num_classes,
        use_thresh=args.use_thresh,
        arch=args.arch,
        pretrained=args.pretrained,
        add_on_layers_type=args.add_on_layers_type,
        prototype_activation_function=args.prototype_activation_function,
        proto_depth=args.proto_depth,
        use_last_layer=args.last_layer,
        inat=args.inat,
        cap_width = cap_width
    )
    if args.ppnet_path:
        model.load_state_dict(torch.load(args.ppnet_path, map_location='cpu')[
                              'model_state_dict'], strict=True)
        print('Successfully loaded ' + args.ppnet_path)

    model.to(device)
    if args.warmup:
        model.features.requires_grad_(False)
        model.last_layer.requires_grad_(True)
        if args.ppnet_path:
            model.add_on_layers.requires_grad_(False)
            model.prototype_vectors.requires_grad_(False)
    if args.checkpoint:
        model, start_epoch = load_model(model, args.checkpoint, device)
    else:
        start_epoch = 0

    warm_optimizer = torch.optim.Adam(
        [{'params': model.add_on_layers.parameters(), 'lr': 3 * args.lr, 'weight_decay': 1e-3},
         {'params': model.proto_presence, 'lr': 3 * args.lr},
         {'params': model.prototype_vectors, 'lr': 3 * args.lr},
         {'params': model.cap_width_l2, 'lr': args.lr/10}]
         
         )
    if args.only_warmuptr:
        joint_optimizer = torch.optim.Adam(
            [{'params': model.features.parameters(), 'lr': args.lr / 10, 'weight_decay': 1e-3},
             {'params': model.add_on_layers.parameters(), 'lr': 3 * args.lr,
              'weight_decay': 1e-3},
             {'params': model.prototype_vectors, 'lr': 3 * args.lr},
             {'params': model.proto_presence, 'lr': 3 * args.lr}]
        )
    else:
        joint_optimizer = torch.optim.Adam(
            [{'params': model.features.parameters(), 'lr': args.lr / 10, 'weight_decay': 1e-3},
             {'params': model.add_on_layers.parameters(), 'lr': 3 * args.lr,
              'weight_decay': 1e-3},
             {'params': model.proto_presence, 'lr': 3 * args.lr},
             {'params': model.prototype_vectors, 'lr': 3 * args.lr},
             {'params': model.cap_width_l2, 'lr': args.lr}
             ])
    
    push_optimizer = torch.optim.Adam(
        [{'params': model.last_layer.parameters(), 'lr': args.lr / 10,
          'weight_decay': 1e-3}, ]
    )
    finetune_optimzier =  torch.optim.Adam(
        [{'params': model.last_layer.parameters(), 'lr': args.lr /5}]
    )

    optimizer = warm_optimizer
    criterion = torch.nn.CrossEntropyLoss()
          
    info = f'capwith_{cap_width}' \
           f'capcoef_{args.capcoef}' \
           f'lr_upby100'\
           f'only_warmup_{args.only_warmuptr}'\
           f'sep_{args.sep}'\
           f'topkclst_{args.topk_loss}'\
           f'cap_all_{args.cap_all}'\
           f'top_k_{top_k}'\
           f'control'
           
    path_tensorboard = f'{args.results}/tensorboard/{info}'
    Path(path_tensorboard).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path_tensorboard)
    dir_checkpoint = f'{args.results}/checkpoint/{info}'
    if args.proto_img_dir:
        proto_img_dir = f'{args.results}/img_proto/{info}'
        Path(proto_img_dir).mkdir(parents=True, exist_ok=True)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    ####################################
    #          learning model          #
    ####################################
    min_val_loss = np.Inf
    max_val_tst = 0
    epochs_no_improve = 0

    epoch_tqdm = range(start_epoch, args.epochs)
    steps = False

    model_multi = torch.nn.DataParallel(model)

    if not args.push_only:
        print('Model learning')
        for epoch in epoch_tqdm:
            gumbel_scalar = lambda1(epoch) if args.pp_gumbel else 0
            if args.warmup and args.warmup_time == epoch:
                model_multi.module.features.requires_grad_(True)
                optimizer = joint_optimizer
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.1)
                steps = True
                print("Warm up ends")

            model_multi.train()
            #if (epoch + 1) % 8 == 0 and tau > 0.3:
                #tau = 0.8 * tau

            ####################################
            #            train step            #
            ####################################
            trn_loss = 0
            trn_tqdm = enumerate(train_loader, 0)
            if epoch > 0:
                for i, (data, label) in trn_tqdm:
                    label_p = label.numpy().tolist()
                    data = data.to(device)
                    label = label.to(device)

                    if args.mixup_data:
                        data, targets_a, targets_b, lam = mixup_data(
                            data, label, 0.5)

                    # ===================forward=====================
                    prob, min_distances, proto_presence, _, _, _ = model_multi(
                        data, gumbel_scale=gumbel_scalar, cap_all = args.cap_all,vis_cap = False)
                    cap_loss = torch.linalg.norm(model_multi.module.cap_width_l2)
                    np.savez_compressed(f'{dir_checkpoint}/pp_{epoch * 80 + i}.pth', proto_presence.detach().cpu().numpy())


                    if args.mixup_data:
                        entropy_loss = lam * \
                            criterion(prob, targets_a) + (1 - lam) * \
                            criterion(prob, targets_b)
                    else:
                        entropy_loss = criterion(prob, label)
                    orthogonal_loss = torch.Tensor([0]).cuda()
                    if args.pp_ortho:
                        for c in range(0, model_multi.module.proto_presence.shape[0], 1000):
                            orthogonal_loss_p = \
                                torch.nn.functional.cosine_similarity(model_multi.module.proto_presence.unsqueeze(2)[c:c+1000],
                                                                      model_multi.module.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            orthogonal_loss += orthogonal_loss_p
                        orthogonal_loss = orthogonal_loss / (args.num_descriptive * args.num_classes) - 1

                    proto_presence = proto_presence[label_p]
                    inverted_proto_presence = 1 - proto_presence
                    clst_loss_val = \
                        dist_loss(model_multi.module, min_distances, proto_presence,
                                  args.num_descriptive, topk_loss = args.topk_loss, k = top_k)  
                    sep_loss_val = dist_loss(model_multi.module, min_distances, inverted_proto_presence,
                                             args.num_prototypes - args.num_descriptive, topk_loss = args.topk_loss,
                                             sep = args.sep)  

                    prototypes_of_correct_class = proto_presence.sum(
                        dim=-1).detach()
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    avg_separation_cost = \
                        torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                                dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)

                    l1_mask = 1 - \
                        torch.t(model_multi.module.prototype_class_identity).cuda()
                    l1 = (model_multi.module.last_layer.weight * l1_mask).norm(p=1)

                    loss = entropy_loss + clst_loss_val * clst_weight + \
                        sep_loss_val * sep_weight + 1e-4 * l1 + orthogonal_loss + args.capcoef*cap_loss

                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('train/loss', loss,
                                      epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/entropy', entropy_loss.item(), epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/clst', clst_loss_val.item(), epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/sep', sep_loss_val.item(), epoch * len(train_loader) + i)
                    writer.add_scalar('train/l1', l1.item(),
                                      epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/avg_sep', avg_separation_cost.item(), epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/orthogonal_loss', orthogonal_loss.item(), epoch * len(train_loader) + i)
                    trn_loss += loss.item()
                    writer.add_scalar('cap loss',cap_loss.item(),epoch*len(train_loader) + i)
                trn_loss /= len(train_loader)
            if steps:
                lr_scheduler.step()

            ####################################
            #          validation step         #
            ####################################
            model_multi.eval()
            tst_loss = np.zeros((args.num_classes, 1))
            prob_leaves = np.zeros((args.num_classes, 1))
            tst_acc, total = 0, 0
            tst_tqdm = enumerate(test_loader, 0)
            with torch.no_grad():
                for i, (data, label) in tst_tqdm:
                    data = data.to(device)
                    label_p = label.detach().numpy().tolist()
                    label = label.to(device)

                    # ===================forward=====================
                    prob, min_distances, proto_presence, _,_,_ = model_multi(data, 
                                                            gumbel_scale=gumbel_scalar, cap_all = args.cap_all,
                                                            vis_cap = False)
                    cap_loss_t = torch.linalg.norm(model_multi.module.cap_width_l2)
                    loss = criterion(prob, label)
                    entropy_loss = loss

                    orthogonal_loss = 0
                    orthogonal_loss = torch.Tensor([0]).cuda()                                                                                                                                            
                    if args.pp_ortho: 
                        for c in range(0, model_multi.module.proto_presence.shape[0], 1000):
                            orthogonal_loss_p = \
                                torch.nn.functional.cosine_similarity(model_multi.module.proto_presence.unsqueeze(2)[c:c+1000],
                                                                      model_multi.module.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            orthogonal_loss += orthogonal_loss_p
                        orthogonal_loss = orthogonal_loss / (args.num_descriptive * args.num_classes) - 1
                    inverted_proto_presence = 1 - proto_presence

                    l1_mask = 1 - torch.t(model_multi.module.prototype_class_identity).cuda()
                    l1 = (model_multi.module.last_layer.weight * l1_mask).norm(p=1)

                    proto_presence = proto_presence[label_p]
                    inverted_proto_presence = inverted_proto_presence[label_p]
                    clst_loss_val = dist_loss(model_multi.module, min_distances, proto_presence, args.num_descriptive,topk_loss = args.topk_loss, k = top_k) * clst_weight
                    sep_loss_val = dist_loss(model_multi.module, min_distances, inverted_proto_presence, args.num_prototypes - args.num_descriptive, topk_loss=args.topk_loss,
                    sep = args.sep) * sep_weight
                    loss = entropy_loss + clst_loss_val + sep_loss_val + orthogonal_loss + 1e-4 * l1 + args.capcoef*cap_loss_t
                    tst_loss += loss.item()

                    _, predicted = torch.max(prob, 1)
                    prob_leaves += prob.mean(dim=0).unsqueeze(
                        1).detach().cpu().numpy()
                    true = label
                    tst_acc += (predicted == true).sum()
                    total += label.size(0)

            tst_loss /= len(test_loader)
            tst_acc = tst_acc.item() / total

            ####################################
            #             logger               #
            ####################################

            tst_loss = tst_loss.mean()
            if trn_loss is None:
                trn_loss = loss.mean().detach()
                trn_loss = trn_loss.cpu().numpy() / len(train_loader)
            print(f'Epoch {epoch}|{args.epochs}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
                  f'| acc: {tst_acc:.5f}, orthogonal: {orthogonal_loss.item():.5f} '
                  f'(minimal test-loss: {min_val_loss:.5f}, early stop: {epochs_no_improve}|{args.earlyStopping}) - ',
                  f'(cap loss: {args.capcoef*cap_loss_t}')
            ####################################
            #  scheduler and early stop step   #
            ####################################
            if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
                # save the best model
                if tst_acc > max_val_tst:
                    save_model(model_multi.module, f'{dir_checkpoint}/best_model_epoch{epoch}.pth', epoch)

                epochs_no_improve = 0
                if tst_loss.mean() < min_val_loss:
                    min_val_loss = tst_loss.mean()
                if tst_acc > max_val_tst:
                    max_val_tst = tst_acc
            else:
                epochs_no_improve += 1

            if args.use_scheduler:
                # scheduler.step()
                if epochs_no_improve > 5:
                    adjust_learning_rate(optimizer, 0.95)

            if args.earlyStopping is not None and epochs_no_improve > args.earlyStopping:
                print('\033[1;31mEarly stopping!\033[0m')
                break
    save_model(model_multi.module, f'{dir_checkpoint}/model_final_epoch{epoch}.pth', epoch)
    ########################### load and vis ##########################
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    model_multi.eval()
    img_dir = args.proto_img_dir
    makedir(img_dir)
    with torch.no_grad():
        vis_count = vis_caps.vis_prototypes(
                caps_anal_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=model_multi, # pytorch network with prototype_vectors
                class_specific=True,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=proto_img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                cap_all = args.cap_all,
                gumbel_scalar = gumbel_scalar)
        # find the prototypes that need to prune
    print('Retrieving the Fully Actitivated Prototypes:')
    print('Starts Prunning and finetuning')
    model_multi.module.prune(vis_count)
    print()
    num = (vis_count > 0).sum()
    print('The estimated number of visualizable prototypes:')
    print(num)
    print()
    ########################################################################
    max_val_tst = 0
    min_val_loss = 10e5
    model_multi.module.fine_tune_last_only()
    for tune_epoch in range(args.fine_tune_epoch):
        model_multi.train()
        trn_loss = 0
        trn_tqdm = enumerate(train_loader, 0)
        for i, (data, label) in trn_tqdm:
            data = data.to(device)
            label = label.to(device)
            finetune_optimzier.zero_grad()
            # ===================forward=====================
            if args.mixup_data:
                data, targets_a, targets_b, lam = mixup_data(data, label, 0.5)
            prob, min_distances, proto_presence,_, _,_ = model_multi(data, 
            gumbel_scale=10e3,cap_all = args.cap_all,vis_cap = False)
            if args.mixup_data:
                entropy_loss = lam * \
                    criterion(prob, targets_a) + (1 - lam) * \
                    criterion(prob, targets_b)
            else:
                entropy_loss = criterion(prob, label)
            l1_mask = 1 - torch.t(model_multi.module.prototype_class_identity).cuda()
            l1 = 1e-4 * (model_multi.module.last_layer.weight * l1_mask).norm(p=1)
            loss = entropy_loss + l1
            loss.backward()
            finetune_optimzier.step()
            trn_loss += loss.item()
        ####################################
        #          validation step         #
        ####################################
        model_multi.eval()
        tst_loss = np.zeros((args.num_classes, 1))
        tst_acc, total = 0, 0
        tst_tqdm = enumerate(test_loader, 0)
        with torch.no_grad():
            for i, (data, label) in tst_tqdm:
                data = data.to(device)
                label = label.to(device)
                # ===================forward=====================
                prob, min_distances, proto_presence, _,_,_= model_multi(data, 
                gumbel_scale=10e3,cap_all = args.cap_all)
                entropy_loss = criterion(prob, label)
                l1_mask = 1 - torch.t(model_multi.module.prototype_class_identity).cuda()
                l1 = 1e-4 * (model_multi.module.last_layer.weight * l1_mask).norm(p=1)
                loss = entropy_loss + l1
                tst_loss += loss.item()
                _, predicted = torch.max(prob, 1)
                tst_acc += (predicted == label).sum()
                total += label.size(0)
            tst_loss /= len(test_loader)
            tst_acc = tst_acc.item() / total
        tst_loss = tst_loss.mean()
        if trn_loss is None:
            trn_loss = loss.mean().detach()
            trn_loss = trn_loss.cpu().numpy() / len(train_loader)
        print(f'Epoch {tune_epoch}|{args.fine_tune_epoch}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
                f'| acc: {tst_acc:.5f}, (minimal test-loss: {min_val_loss:.5f}- ')
        ####################################
        #  scheduler and early stop step   #
        ####################################
        if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
            # save the best model
            if tst_acc > max_val_tst:
                save_model(model_multi.module, f'{dir_checkpoint}/best_model_finetune_epoch{tune_epoch}.pth', tune_epoch)
                max_val_tst = tst_acc
            if tst_loss.mean() < min_val_loss:
                min_val_loss = tst_loss.mean()
        
        if (tune_epoch + 1) % 5 == 0:
            adjust_learning_rate(finetune_optimzier, 0.95)
        
    writer.close()
    print('Finished training model. Have nice day :)')


def dist_loss(model, min_distances, proto_presence, top_k, topk_loss=False, sep = False, k = 10):
    #         model, [b, p],        [b, p, n],      [scalar]
    max_dist = (model.prototype_shape[1]
                * model.prototype_shape[2]
                * model.prototype_shape[3])
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]
    _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(1, src=torch.ones_like(
        basic_proto), index=idx)  # [b, p]
    if not topk_loss or sep:
        inverted_distances, _ = torch.max(
            (max_dist - min_distances) * binarized_top_k, dim=1)  # [b]
    elif topk_loss:
        inverted_distances, _ = torch.topk(
        (max_dist - min_distances) * binarized_top_k, k, dim=1)  # [b]
    cost = torch.mean(max_dist - inverted_distances)
    return cost



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrototypeGraph')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='The run evaluation training model')
    args, unknown = parser.parse_known_args()

    learn_model(unknown)

