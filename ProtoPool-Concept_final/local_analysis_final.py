import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.autograd import Variable
import pickle
from model_cap_final import Protopool_cap
from utils import mixup_data, find_high_activation_crop
import os
import matplotlib.pyplot as plt
import cv2
from utils import mixup_data, compute_proto_layer_rf_info_v2, compute_rf_prototype
from PIL import Image
import copy
from preprocess import preprocess_input_function, undo_preprocess_input_function
import pickle

##### HELPER FUNCTIONS FOR PLOTTING
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname,img_dir):
    p_img = plt.imread(img_dir)
    plt.imsave(fname, p_img)

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)

def load_model(model, path, device):
    if device.type == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'\033[0;32mLoad model form: {path}\033[0m')
    return model, checkpoint['epoch']


def local_analysis(imgs, model_multi, proto_info_dir, vis_count,
                    save_analysis_dir, test_data, 
                    transforms_train_test,prototype_img_identity,protoc_info):
    imgs_sep = imgs.split('/') # eg. 083.White_breasted_Kingfisher\White_Breasted_Kingfisher_0012_73367.jpg
    img_file_name = imgs_sep[0] # eg. 083.White_breasted_Kingfisher
    #eg. ~./083.White_breasted_Kingfisher/White_Breasted_Kingfisher_0012_73367.jpg
    analysis_rt = os.path.join(save_analysis_dir, imgs_sep[0],imgs_sep[1]) 
    makedir(analysis_rt)
    img_rt = os.path.join(test_data, imgs)
    img_pil = Image.open(img_rt)
    img_tensor = transforms_train_test(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.cuda()
    test_image_label = int(img_file_name[0:3])-1 # lable is on 0 idx based 
    labels_test = torch.tensor([test_image_label])
    prob, min_dist, proto_presence, cap_factor, avg_act, proto_act_capped = model_multi(images_test,
                                            gumbel_scale = 10e3,
                                            vis_cap = False)
    cap_factor_vis,big_avg_act, big_proto_act_capped = model_multi(images_test,
                                            gumbel_scale = 10e3,
                                            vis_cap = True)
    predicted = torch.argmax(prob, dim=1).item()
    print('The predicted label is ', predicted)
    print('The actual lable is', labels_test.item())
    idx = 0 
    original_img = save_preprocessed_img(os.path.join(analysis_rt, 'original_img.png'),
                                     img_variable, idx)
    # ######################### Most activated prototypes ##########################################
    most_activated_rt = os.path.join(analysis_rt, 'most_activated_prototypes')
    makedir(most_activated_rt)
    print('Most activated 10 prototypes of this image:')
    # by focal similarity
    act, identity = (proto_act_capped-avg_act).topk(10)
    identity = identity.cpu().detach().numpy()[0]
    print(identity)
    print('Prototypes selected for predicted class after pruned')
    # trivial for pnet and tesnet 
    # the selected prototypes q distribution for predicted class 
    protoprec_pred = proto_presence[predicted].cpu().detach().numpy()
    protoprec_sel =np.argwhere(protoprec_pred)[:,0]
    print(protoprec_sel)
    print('the given slosts by prediction')
    slots= np.argwhere(protoprec_pred)[:,1]
    print(slots)
    llw = model_multi.module.last_layer.weight.reshape(200, 200, 10)
    prototype_img_filename_prefix='prototype-img'
    for idx, i in enumerate(identity):
        # check if there is visualization for given prototypes
        if vis_count[i] != 0:
            print('prototype index: {0}'.format(i))
            print('prototype class identity: {0}'.format(prototype_img_identity[str(i)]))
            print('activation value (similarity score): {0}'.format(act[0][idx]))
            # check if the most activated prototype contributes to the prediction 
            if i in protoprec_sel:
                idx_slot = np.where(protoprec_sel == i)
                print('idx_slot',idx_slot)
                for slot_i in range(len(idx_slot[0])):
                    idx_sl = idx_slot[0][slot_i]
                    slot = slots[idx_sl]
                    print(llw[predicted][predicted][slot])
                    print('last layer connection with predicted class: {0}'.format(llw[predicted][predicted][slot].item()))
            elif i not in protoprec_sel:
                # when activation is not contributing to the correct class
                print(f'{i}prototype does not contribute to the largest prob')
                contributed_idxes = np.argwhere(proto_presence[:, i, :].detach().cpu().numpy())
                for contributed_idx in contributed_idxes:
                    contributed_class = contributed_idx[0]
                    contributed_slot = contributed_idx[1]
                    print(f'last layer connection to other classes {contributed_class}: {llw[contributed_class][contributed_class][contributed_slot]}')
            # all the visualizations for a given prototypes 
            try:
                for j in range(1,vis_count[i]+1):
                    vis_ci = protoc_info[str(i)][j-1]
                    #save prototypes to analysis file
                    proto_dir = os.path.join(proto_info_dir, prototype_img_filename_prefix + str(i) + '_' + str(j) + '.png')
                    saved_proto_dir = os.path.join(most_activated_rt, 'top-%d_activated_prototype_%d_%d.png'%(idx+1,vis_ci,j))
                    save_prototype(saved_proto_dir,proto_dir)
                    # save prototype self activation 
                    proto_activ_dir = os.path.join(proto_info_dir, 'prototype-img-original_with_self_act' + str(i) + '_' + str(j) + '.png')
                    saved_proto_activ_dir = os.path.join(most_activated_rt, 'top-%d_activated_prototype_self_act_%d_%d.png'%(idx+1,vis_ci,j))
                    save_prototype(saved_proto_activ_dir,proto_activ_dir)
                    #save proto_org_bb to file
                    bb_dir = os.path.join(proto_info_dir, prototype_img_filename_prefix + 'bbox-original' + str(i) + '_' + str(j) + '.png')
                    saved_bb_dir = os.path.join(most_activated_rt, 'top-%d_activated_prototype_in_original_pimg_%d_%d.png'%(idx+1,vis_ci,j))
                    save_prototype(saved_bb_dir,bb_dir)
                # create activation of test img for each prototypes 
                activation_pattern = (big_proto_act_capped-big_avg_act)[0][i].detach().cpu().numpy()
                upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(224, 224),
                                                        interpolation=cv2.INTER_CUBIC)
                high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                            high_act_patch_indices[2]:high_act_patch_indices[3], :]
                plt.imsave(os.path.join(most_activated_rt,
                                        'most_highly_activated_patch_by_-%d_prototype.png' % (idx+1)),
                        high_act_patch)
                imsave_with_bbox(fname=os.path.join(most_activated_rt,
                                        'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % (idx+1)),
                                img_rgb=original_img,
                                bbox_height_start=high_act_patch_indices[0],
                                bbox_height_end=high_act_patch_indices[1],
                                bbox_width_start=high_act_patch_indices[2],
                                bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
                # show the image overlayed with prototype activation map
                upsampled_activation_pattern = np.exp(upsampled_activation_pattern)
                rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_img = 0.5 * original_img + 0.3 * heatmap
                print('prototype activation map of the chosen image:')
                #plt.axis('off')
                plt.imsave(os.path.join(most_activated_rt,
                                        'prototype_activation_map_by_top-%d_prototype.png' % (idx+1)),
                        overlayed_img, vmin = 0.0 , vmax = 1.0)
            except:
                pass
    ##### PROTOTYPES FROM TOP-k CLASSES
    k = 5
    print('Prototypes from top-%d classes:' % k)
    topk_prob, topk_classes = torch.topk(prob[0], k=k)
    for i,c in enumerate(topk_classes.detach().cpu().numpy()):
        topk_dir = os.path.join(analysis_rt, f'top-{i+1}_class_prototypes_class{c+1}')
        makedir(topk_dir)
        print('top %d predicted class: %d' % (i+1, c))
        print('logit of the class: %f' % topk_prob[i])
        protoprec_k = proto_presence[c].cpu().detach().numpy()
        protoprec_selk =np.argwhere(protoprec_k)[:,0]
        print(protoprec_selk)
        slots_k= np.argwhere(protoprec_k)[:,1]
        prototype_img_filename_prefix='prototype-img'
        act = (proto_act_capped-avg_act)
        _, order = torch.sort(act[0][protoprec_selk], descending = True)
        for idx, k in enumerate(order):
            # check if there is visualization for given prototypes
            # i: proto idx 
            i = protoprec_selk[k]
            if vis_count[i] != 0:
                print('prototype index: {0}'.format(i))
                print('prototype class identity: {0}'.format(prototype_img_identity[str(i)]))
                print('activation value (similarity score): {0}'.format(act[0][i]))
                slot_k = slots_k[k]
                print('last layer connection with predicted class: {0}'.format(llw[c][c][slot_k]))
                try:
                    for j in range(1,vis_count[i]+1):
                        vis_ci = protoc_info[str(i)][j-1]+1
                        #save prototypes to analysis file
                        proto_dir = os.path.join(proto_info_dir, prototype_img_filename_prefix + str(i) + '_' + str(j) + '.png')
                        saved_proto_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_%d_%d.png'%(idx+1,vis_ci,j))
                        save_prototype(saved_proto_dir,proto_dir)
                        # save prototype self activation 
                        proto_activ_dir = os.path.join(proto_info_dir, 'prototype-img-original_with_self_act' + str(i) + '_' + str(j) + '.png')
                        saved_proto_activ_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_self_act_%d_%d.png'%(idx+1,vis_ci,j))
                        save_prototype(saved_proto_activ_dir,proto_activ_dir)
                        #save proto_org_bb to file
                        bb_dir = os.path.join(proto_info_dir, prototype_img_filename_prefix + 'bbox-original' + str(i) + '_' + str(j) + '.png')
                        saved_bb_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_in_original_pimg__%d_%d.png'%(idx+1,vis_ci,j))
                        save_prototype(saved_bb_dir,bb_dir)

                    # create activation of test img for each prototypes 
                    activation_pattern = (big_proto_act_capped-big_avg_act)[0][i].detach().cpu().numpy()
                    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(224, 224),
                                                            interpolation=cv2.INTER_CUBIC)
                    upsampled_activation_pattern = np.exp(upsampled_activation_pattern)
                    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                                high_act_patch_indices[2]:high_act_patch_indices[3], :]
                    plt.imsave(os.path.join(topk_dir,
                                            'most_highly_activated_patch_by_-top%d_class.png' %(idx+1)),
                            high_act_patch)
                    imsave_with_bbox(fname=os.path.join(topk_dir,
                                            'most_highly_activated_patch_in_original_img_by_top-%d_class.png' %(idx+1)),
                                    img_rgb=original_img,
                                    bbox_height_start=high_act_patch_indices[0],
                                    bbox_height_end=high_act_patch_indices[1],
                                    bbox_width_start=high_act_patch_indices[2],
                                    bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

                    # show the image overlayed with prototype activation map
                    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_img = 0.5 * original_img + 0.3 * heatmap
                    print('prototype activation map of the chosen image:')
                    plt.imsave(os.path.join(topk_dir,
                                            'prototype_activation_map_by_top-%d_class.png'%(idx)),
                            overlayed_img, vmin = 0.0 , vmax = 1.0)
                except:
                    pass
    return None 

# main function to run local analysis 
def analyze(opt: Optional[List[str]]) -> None:
    parser = argparse.ArgumentParser(description='PrototypeVis')
    ### setting 
    parser.add_argument('--data_train', help='Path to train data')
    parser.add_argument('--data_push', help='Path to push data')
    parser.add_argument('--data_test', help='Path to tets data')
    parser.add_argument('--model_dir', help='saved model directory',type=str)
    parser.add_argument('--batch_size', type=int, default=80,
                        help='input batch size for training (default: 80)')
    ### model params
    parser.add_argument('--num_descriptive', type=int, default=10)
    parser.add_argument('--num_prototypes', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--arch', type=str, default='resnet34')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')
    parser.add_argument('--prototype_activation_function',
                        type=str, default='log')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--use_thresh', action='store_true')
    parser.add_argument('--proto_depth', default=128, type=int)
    parser.add_argument('--last_layer', action='store_true')
    parser.add_argument('--inat', action='store_true')
    parser.add_argument('--gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    parser.add_argument('--capl',default = 0, type = float)
    parser.add_argument('--drop_last',default = True)   
    parser.add_argument('--cap_all', type = bool, default = False)
    ### analysis params 
    parser.add_argument('--save_analysis_dir',type = str, help = 'Path for saving analysis result') 
    parser.add_argument('--target_img_dir', type = str, help ='target image dir rt(test images dir)')
    parser.add_argument('--prototype_img_dir', type = str, help = 'prototype vis dir')
    parser.add_argument('--test_acc', type = bool, default = False)

    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')

    kwargs = {}
    if args.seed is None:  # 1234
        args.seed = np.random.randint(10, 10000, size=1)[0]
        #args.seed = 3407
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        kwargs.update({'num_workers': 2, 'pin_memory': True})
    
    ########################### load data #################
    #######################################################

    transforms_train_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        test_dataset, batch_size=1, shuffle=False, drop_last=False,
        **kwargs)

    push_dataset = datasets.ImageFolder(
    args.data_push,transforms_train_test)

    caps_push_loader = torch.utils.data.DataLoader(
        push_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    print('push set size: {0}'.format(len(caps_push_loader.dataset)))
    print('batch size: {0}'.format(args.batch_size))

    #############################################
    ##### load model ############################

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
        cap_width = args.capl
    )
    model.to(device)
    model, epoch = load_model(model, args.model_dir, device)
    model_multi = torch.nn.DataParallel(model)
    print(f'Successfully loaded model with epoch {epoch}')
    print(device)
    model_multi.eval()
    # retrieve stats data
    proto_info_dir = args.prototype_img_dir
    vis_count_dir = os.path.join(proto_info_dir,'vis_count'+str(29)+'.npy')
    vis_count = np.load(vis_count_dir)
    # retrieve prototype class information
    # contains the class information for each prototype vis 
    # important for protopool and trivial for protopnet and tesnet 
    bb_tr = os.path.join(proto_info_dir,'bb'+str(29)+'ci.pkl') # 'bb29ci.pkl'
    f = open(bb_tr, 'rb')
    protoc_info = pickle.load(f)
    f.close()
    prototype_img_identity = dict()
    for i in protoc_info:
        prototype_img_identity[i] = set(protoc_info[i]) # distinct class for each prototype
    cls = set()
    for i in prototype_img_identity:
        cls.update(prototype_img_identity[i])
    print('Prototypes are chosen from ' +str(len(cls))+' number of classes.')
    non_selected_cls = []
    if len(cls) != args.num_classes:
        for cl in range(args.num_classes):
            if cl not in cls:
                non_selected_cls.append(cl)
        print('The classes not chosen by the prototypes are:', non_selected_cls)
    prototype_max_connection = torch.argmax(model_multi.module.last_layer.weight, dim=0)
    # I don't think this is necessary for protopool 
    if len(prototype_max_connection.unique()) == 200:
        print('All prototypes connect most strongly to their respective classes.')
    else:
        print('WARNING: Not all prototypes connect most strongly to their respective classes.')
    
    # run analysis and retrieve the stats data 
    gumbel_scalar = 10e3
    #test accuracy 
    if args.test_acc:
        model_multi.eval()
        prob_leaves = np.zeros((200, 1))
        tst_acc, total = 0, 0
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader, 0):
                data = data.to(device)
                label = label.to(device)
                prob_tst, _, _, _, _, _ = model_multi(data,
                                                gumbel_scale = 10e3,
                                                vis_cap = False)
                _, predicted_tst = torch.max(prob_tst, 1)
                prob_leaves += prob_tst.mean(dim=0).unsqueeze(1).detach().cpu().numpy()
                true = label
                tst_acc += (predicted_tst == true).sum()
                total += label.size(0)
            tst_acc = tst_acc.item() / total
            print('The testing accuracy of the given model is:',tst_acc)
    # just an example, I leave this in for easier modification to run analysis on all the images 
    img_list = ['009.Brewer_Blackbird/Brewer_Blackbird_0004_2345.jpg']
    save_analysis_dir = args.save_analysis_dir
    for img in img_list:
        # note that there are images that are only in gray scale from the test-set. 
        #try:
        print('run local analysis for:',img)
        local_analysis(img, model_multi, 
                        proto_info_dir, vis_count, 
                        save_analysis_dir,args.data_test,
                        transforms_train_test,
                        prototype_img_identity,protoc_info)
        #except:
            #pass
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prototype_local_analysis')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='The run evaluation training model')
    args, unknown = parser.parse_known_args()

    analyze(unknown)


