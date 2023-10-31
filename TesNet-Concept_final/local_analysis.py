##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import os
import shutil
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import argparse
import re
import model_cap
#import train_and_test_ctrl_caps as tnt
from pathlib import Path
from util.helpers import makedir, find_high_activation_crop
import train_and_test_cap as tnt
from util.log import create_logger
from util.preprocess import mean, std, undo_preprocess_input_function
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import copy
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

def local_analysis(imgs, ppnet_multi,protoc_info, save_analysis_path,test_image_dir,
                        prototype_img_identity,
                        start_epoch_number,
                        load_img_dir,
                        vis_count):
    
    # only run top1 class 
    imgs_sep = imgs.split('/') # eg. 083.White_breasted_Kingfisher\White_Breasted_Kingfisher_0012_73367.jpg
    img_file_name = imgs_sep[0] # eg. 083.White_breasted_Kingfisher
    #img_name = imgs_sep[1] # eg. White_Breasted_Kingfisher_0012_73367.jpg
    analysis_rt = os.path.join(save_analysis_path, imgs_sep[0], imgs_sep[1])# dir to save the analysis class 
    makedir(analysis_rt)
    img_size = ppnet_multi.module.img_size
    normalize = transforms.Normalize(mean=mean,
                                    std=std)
    preprocess = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    normalize
    ])
    img_rt = os.path.join(test_image_dir, imgs)
    img_pil = Image.open(img_rt)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.cuda()
    test_image_label = int(img_file_name[0:3])-1
    labels_test = torch.tensor([test_image_label])
    logits, cosine_min_distances,prototype_activations, cap_factor = ppnet_multi(images_test,vis_cap=False)
    #prototype_activations, cap_factor = ppnet_multi(images_test,get_act=True, vis_cap=False, stats =stats_count)
    big_proto_act_capped, cap_factor = ppnet_multi(images_test,vis_cap=True)
    _, predicted = torch.max(logits.data, 1)
    print('The predicted label is ', predicted)
    print('The actual lable is', labels_test.item())
    #idx = 0 
    original_img = save_preprocessed_img(os.path.join(analysis_rt, 'original_img.png'),
                                     img_variable, index = 0 )
    ##### PROTOTYPES FROM TOP-k CLASSES
    k = 5
    print('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[0], k=k)
    prototype_img_filename_prefix='prototype-img'
    for idx,c in enumerate(topk_classes.detach().cpu().numpy()):
        topk_dir = os.path.join(analysis_rt, 'top-%d_class_prototypes_class%d' % ((idx+1),c+1))
        makedir(topk_dir)
        print('top %d predicted class: %d' % (idx+1, c+1))
        print('logit of the class: %f' % topk_logits[idx])
        class_prototype_indices = np.nonzero(ppnet_multi.module.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[0][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)
        print('class prototype activations:',class_prototype_activations)
        iterat = 0 
        for s in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            #print('class prototype activations:',class_prototype_activations)
            prototype_index = class_prototype_indices[s]
            if vis_count[prototype_index] != 0:
                print('prototype index: {0}'.format(prototype_index))
                print('prototype class identity: {0}'.format(prototype_img_identity[str(prototype_index)]))
                print('activation value (similarity score): {0}'.format(class_prototype_activations[s]))
                for j in range(1,vis_count[prototype_index]+1):
                    try:
                        vis_ci = protoc_info[str(prototype_index)][j-1]+1
                        proto_dir = os.path.join(load_img_dir, prototype_img_filename_prefix + str(prototype_index) + '_' + str(j) + '.png')
                        saved_proto_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_%d_%d.png'%(iterat+1,vis_ci,j))
                        save_prototype(saved_proto_dir,proto_dir)
                        bb_dir = os.path.join(load_img_dir, prototype_img_filename_prefix + 'bbox-original' + str(prototype_index) + '_' + str(j) + '.png')
                        saved_bb_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_in_original_pimg__%d_%d.png'%(iterat+1,vis_ci,j))
                        save_prototype(saved_bb_dir,bb_dir)
                    except:
                        pass

                activation_pattern = big_proto_act_capped[0][prototype_index].detach().cpu().numpy()
                upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(224, 224),
                                                        interpolation=cv2.INTER_CUBIC)
                upsampled_activation_pattern = np.exp(upsampled_activation_pattern)
                high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                            high_act_patch_indices[2]:high_act_patch_indices[3], :]
                plt.imsave(os.path.join(topk_dir,
                                        'most_highly_activated_patch_by_-top%d_class.png' %(iterat+1)),
                        high_act_patch)
                imsave_with_bbox(fname=os.path.join(topk_dir,
                                        'most_highly_activated_patch_in_original_img_by_top-%d_class.png' %(iterat+1)),
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
                                        'prototype_activation_map_by_top-%d_class.png'%(iterat+1)),
                        overlayed_img, vmin = 0.0 , vmax = 1.0)
            iterat+=1

    return None


def analyze(opt: Optional[List[str]]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--modeldir', nargs=1, type=str)
    parser.add_argument('--model', nargs=1, type=str)
    parser.add_argument('--save_analysis_dir',type = str, help = 'Path for saving analysis result') 
    parser.add_argument('--test_dir',type = str)
    parser.add_argument('-imgdir', nargs=1, type=str)
    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    test_image_dir = args.test_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')
    kwargs = {}
    # load the model
    
    #log, logclose = create_logger(log_filename=os.path.join(args.save_analysis_dir, 'local_analysis.log'))
    check_test_accu = True
    load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
    load_model_name = args.model[0] #'10_18push0.7822.pth'
    model_base_architecture = load_model_dir.split('/')[2]
    experiment_run = '/'.join(load_model_dir.split('/')[3:])

    save_analysis_path = args.save_analysis_dir
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(args.save_analysis_dir, 'local_analysis.log'))
    # load the model
    check_test_accu = False
    model_base_architecture = load_model_dir.split('/')[-3]
    experiment_run = load_model_dir.split('/')[-2]
    load_model_path = os.path.join(load_model_dir, load_model_name)
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)
    print('load model from ' + load_model_path)
    print('model base architecture: ' + model_base_architecture)
    print('experiment run: ' + experiment_run)
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    class_specific = True
    normalize = transforms.Normalize(mean=mean,
                                 std=std)
    # load the test data and check test accuracy
    # cap analysis set (push with normalize)
    from settings_CUB import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size
    train_push_batch_size = 50
    ##### SANITY CHECK
    # confirm prototype class identity
    load_img_dir = os.path.join(load_model_dir, 'img')
    ## load vis_count 
    vis_count_dir = os.path.join(load_img_dir,'vis_count.npy')
    vis_count = np.load(vis_count_dir)
    bb_tr = os.path.join(load_img_dir,'bbci.pkl')
    f = open(bb_tr, 'rb')
    protoc_info = pickle.load(f) # saves the class idx for each prototype vis
    f.close()
    prototype_img_identity = dict()
    for i in protoc_info:
        prototype_img_identity[i] = set(protoc_info[i]) # distinct class for each prototype
    cls = set()
    for i in prototype_img_identity:
        cls.update(prototype_img_identity[i])
    print('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    #print('Their class identities are: ' + str(prototype_img_identity))

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        print('All prototypes connect most strongly to their respective classes.')
    else:
        print('WARNING: Not all prototypes connect most strongly to their respective classes.')
    check_test_accu = False
    if check_test_accu:
        #test_batch_size = 100

        # test set
        from settings_CUB import test_batch_size
        
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        print('test set size: {0}'.format(len(test_loader.dataset)))

        accu= tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=True, log=log)
        print('the accuracy of the model is:',accu)
        
    ##################
    #["068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0131_57813.jpg"]
    list2 = ['074.Florida_Jay/Florida_Jay_0012_64887.jpg',
              '149.Brown_Thrasher/Brown_Thrasher_0042_155213.jpg',
              '186.Cedar_Waxwing/Cedar_Waxwing_0001_179170.jpg',
              '165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0008_164001.jpg']
    for name in list2:
        local_analysis(name, ppnet_multi,
                        protoc_info, 
                        save_analysis_path,test_image_dir,
                        prototype_img_identity,
                        start_epoch_number,
                        load_img_dir,
                        vis_count)
        count +=1 
                 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prototype_local_analysis')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='The run evaluation training model')
    args, unknown = parser.parse_known_args()

    analyze(unknown)