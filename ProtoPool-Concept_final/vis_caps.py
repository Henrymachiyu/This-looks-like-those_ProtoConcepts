import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
import pdb
import math
import torch.nn.functional as F
from utils import mixup_data, compute_proto_layer_rf_info_v2, compute_rf_prototype, find_high_activation_crop
import pickle 

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def make_correct_class(idx, proto_correct_class):
    result = proto_correct_class.clone()
    for i in range(idx.shape[0]):
        idxs = idx[i,:]
        result[i,idxs] += 1
    return result.cuda()

def upscale_rf(protoL_rf_info, proto_act_img_j, img_size): 
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = math.floor(protoL_rf_info[3])

    x_min = start
    x_max = start + (proto_act_img_j.size(0) - 1) * j

    mesh = torch.tensor([-1 + i * 2. / (j * (proto_act_img_j.size(0) - 1)) for i in range(img_size)])
    mesh_x, mesh_y = torch.meshgrid(mesh, mesh, indexing='ij')
    grid = torch.stack((mesh_y, mesh_x), dim=2).unsqueeze(0)
    gridsample_in = proto_act_img_j.unsqueeze(0).unsqueeze(1)
    upsampled_act_img_j = F.grid_sample(gridsample_in, grid, mode='bicubic', padding_mode='zeros', align_corners=True)

    return upsampled_act_img_j

def save_prototype_original_img_with_bbox(save_dir, img_dir,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(img_dir)
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb,vmin=0.0,vmax=1.0)

def vis_cap(model, x, cap_all):
    distances = model.module.prototype_distances(x)
    big_proto_act = model.module.distance_2_similarity(distances)
    cap_factor = model.module.cap_activation(model.module.cap_width_l2).unsqueeze(2).unsqueeze(3)
    if cap_all: # cap on the avg act and proto act
        avg_d = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                            distances.size()[3]))
        avg_act = model.module.distance_2_similarity(avg_d)
        #print('avg_act shape:', avg_act.shape)
        big_proto_act = big_proto_act - avg_act 
        big_proto_act = model.module.soft_clamp(big_proto_act+cap_factor, val= model.module.max_act_l2, 
            upper = True) - cap_factor.detach()
    else:
        big_proto_act = model.module.soft_clamp(big_proto_act + cap_factor, val=model.module.max_act_l2, upper=True) \
                                        - cap_factor.detach()
    return cap_factor, big_proto_act


def update_prototypes_on_batch(search_batch_input, 
                            start_index_of_search_batch,
                            model,
                            proto_rf_boxes,  # this will be updated
                            proto_bound_boxes,  # this will be updated
                            search_y=None,  # required if class_specific == True
                            class_specific=None,
                            preprocess_input_function=None, # normalize function
                            prototype_layer_stride=1,
                            dir_for_saving_prototypes=None,
                            prototype_img_filename_prefix=None,
                            prototype_self_act_filename_prefix=None,
                            prototype_activation_function_in_numpy=None,
                            cap_all = False,
                            gumbel_scalar = 10e3,
                            vis_count = None,
                            prototype_ci = dict()):
    model.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input
    
    with torch.no_grad():
        search_batch = search_batch.cuda()
        cap_factor_torch, proto_act_torch, proto_act_torch_capped = model(search_batch, gumbel_scalar,cap_all = cap_all, vis_cap = True)
        max_act_l2 = model.module.max_act_l2
        if torch.max(proto_act_torch_capped) >= max_act_l2: 
            print('Warning: Maximum cap factor exceeds maximum possible activation')
        # find fully activated patches
        label_p = search_y.numpy().tolist()
        proto_correct_class = torch.zeros(proto_act_torch.shape)
        map_class_to_prototypes = model.module.get_map_class_to_prototypes()
        map_class_to_prototypes_cc = map_class_to_prototypes[label_p]
        prototypes_of_correct_class = make_correct_class(map_class_to_prototypes_cc, proto_correct_class)
        proto_act_correct_class = proto_act_torch_capped * prototypes_of_correct_class
        ############
        fully_activated_patches = (proto_act_correct_class > torch.clamp(max_act_l2 - cap_factor_torch, min=0))

    # setting for fully activated patches visualization
    proto_act_ = np.copy(proto_act_torch_capped.detach().cpu().numpy())
    full_act_patches_ = np.copy(fully_activated_patches.detach().cpu().numpy())
    patches_to_vis = np.argwhere(full_act_patches_)

    if not hasattr(model, 'tau'): 
        del label_p, prototypes_of_correct_class, proto_act_correct_class
    prototype_shape = model.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    #vis_count = [0 for _ in range(n_prototypes)]
    patch_rep_check = set()
    # only visualize the fully activated patches should follow the algo in pool
    for i in range(len(patches_to_vis)):
        patch_coords = patches_to_vis[i]
        j = patch_coords[1]
        batch_argmin_proto_dist_j = patch_coords[np.arange(len(patch_coords))!=1]
        # get the receptive field boundary of the image patch
        # that generates the representation
        layer_filter_sizes, layer_strides, layer_paddings = model.module.features.conv_info()
        protoL_rf_info = compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings,
                                           prototype_kernel_size=1)
        rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
        
        # get the whole image
        original_img_j = search_batch_input[rf_prototype_j[0]]
        original_img_j = original_img_j.numpy()
        original_img_j = np.transpose(original_img_j, (1, 2, 0))
        original_img_size = original_img_j.shape[0]
        original_img_j = (original_img_j - np.min(original_img_j)) / np.max(original_img_j - np.min(original_img_j))

        # crop out the receptive field
        rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                    rf_prototype_j[3]:rf_prototype_j[4], :]
        
        # save the prototype receptive field information
        proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
        proto_rf_boxes[j, 1] = rf_prototype_j[1]
        proto_rf_boxes[j, 2] = rf_prototype_j[2]
        proto_rf_boxes[j, 3] = rf_prototype_j[3]
        proto_rf_boxes[j, 4] = rf_prototype_j[4]
        if proto_rf_boxes.shape[1] == 6 and search_y is not None:
            proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

        # find the highly activated region of the original image
        proto_act_img_j = proto_act_[patch_coords[0]][patch_coords[1]]
        #upsampled_act_img_j = upscale_rf(protoL_rf_info, torch.tensor(proto_act_img_j), 
                                       # original_img_size)
        upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
        upsampled_act_img_j = upsampled_act_img_j.squeeze()#.numpy()
        proto_bound_j = find_high_activation_crop(upsampled_act_img_j)

        # crop out the image patch with high activation as prototype image
        proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]
        
        # save the prototype boundary (rectangular boundary of highly activated region)
        proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
        proto_bound_boxes[j, 1] = proto_bound_j[0]
        proto_bound_boxes[j, 2] = proto_bound_j[1]
        proto_bound_boxes[j, 3] = proto_bound_j[2]
        proto_bound_boxes[j, 4] = proto_bound_j[3]
        if proto_bound_boxes.shape[1] == 6 and search_y is not None:
            proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()
            value = prototype_ci.get(str(j),[])
            value.append(search_y[rf_prototype_j[0]].item()) # destructive operation 
            prototype_ci[str(j)] = value

        # img code = jth proto _ imgidx _ bb1234
        bbid = str(proto_bound_boxes[j, 1]) + str(proto_bound_boxes[j, 2]) + str(proto_bound_boxes[j, 3]) + str(proto_bound_boxes[j, 4]) 
        img_code = str(j) + "_" + str(rf_prototype_j[0]) + "_" + bbid
        # to avoid the visualization by close patches activation 
        if dir_for_saving_prototypes is not None and img_code not in patch_rep_check:
            vis_count[j] += 1
            if prototype_self_act_filename_prefix is not None:
                # save the numpy array of the prototype self activation
                np.save(os.path.join(dir_for_saving_prototypes,
                                     prototype_self_act_filename_prefix + str(j) + '_' + str(vis_count[j]) + '.npy'),
                        proto_act_img_j)
            if prototype_img_filename_prefix is not None:
                # save the whole image containing the prototype as png
                plt.imsave(os.path.join(dir_for_saving_prototypes,
                                        prototype_img_filename_prefix + '-original' + str(j) + '_' + str(vis_count[j]) + '.png'),
                           original_img_j,
                           vmin=0.0,
                           vmax=1.0)
                rt = os.path.join(dir_for_saving_prototypes,
                        prototype_img_filename_prefix + 'bbox-original' + str(j) + '_' + str(vis_count[j]) + '.png')
                original_img_path = os.path.join(dir_for_saving_prototypes,
                                        prototype_img_filename_prefix + '-original' + str(j) + '_' + str(vis_count[j]) + '.png')
                save_prototype_original_img_with_bbox(rt, original_img_path,
                                          bbox_height_start = proto_bound_j[0], 
                                          bbox_height_end = proto_bound_j[1],
                                          bbox_width_start = proto_bound_j[2], 
                                          bbox_width_end = proto_bound_j[3], color=(0, 255, 255))
                # overlay (upsampled) self activation on original image and save the result
                upsampled_act_img_j = np.exp(upsampled_act_img_j)
                rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                plt.imsave(os.path.join(dir_for_saving_prototypes,
                                        prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '_' + str(vis_count[j]) + '.png'),
                           overlayed_original_img_j,
                           vmin=0.0,
                           vmax=1.0)
                
                # if different from the original (whole) image, save the prototype receptive field as png
                if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-receptive_field' + str(j) + '_' + str(vis_count[j]) + '.png'),
                               rf_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                  rf_prototype_j[3]:rf_prototype_j[4]]
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '_' + str(vis_count[j]) + '.png'),
                               overlayed_rf_img_j,
                               vmin=0.0,
                               vmax=1.0)
                # save the prototype image (highly activated region of the whole image)
                plt.imsave(os.path.join(dir_for_saving_prototypes,
                                        prototype_img_filename_prefix + str(j) + '_' + str(vis_count[j]) + '.png'),
                           proto_img_j,
                           vmin=0.0,
                           vmax=1.0)
        
        patch_rep_check.add(img_code)
    return vis_count

# push each prototype to the nearest patch in the training set
def vis_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    prototype_activation_function_in_numpy=None,
                    cap_all = False,
                    gumbel_scalar = 10e3):
    prototype_network_parallel.eval()
    print('\tvis')
    start = time.time()

    #### rf boxes & bound boxes info setup 
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes,6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes,6],
                                            fill_value=-1)
        prototype_ci = dict()
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)
    ########## prototype dir set up 
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None
    
    ####################
    search_batch_size = dataloader.batch_size    
    vis_count = [0 for _ in range(n_prototypes)]
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        start_index_of_search_batch = push_iter * search_batch_size
        vis_count = update_prototypes_on_batch(search_batch_input=search_batch_input, 
                            start_index_of_search_batch=start_index_of_search_batch,
                            model=prototype_network_parallel,
                            proto_rf_boxes=proto_rf_boxes,
                            proto_bound_boxes=proto_bound_boxes,
                            class_specific=True,
                            search_y=search_y,
                            prototype_layer_stride=1,
                            dir_for_saving_prototypes=proto_epoch_dir,
                            prototype_img_filename_prefix='prototype-img',
                            prototype_self_act_filename_prefix='prototype-self-act',
                            prototype_activation_function_in_numpy=None,
                            cap_all = cap_all,
                            gumbel_scalar = gumbel_scalar,
                            vis_count= vis_count.copy(),
                            prototype_ci = prototype_ci)

    vis_count = np.array(vis_count)                       
    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:

        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)
        root = os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number)+ 'ci'+ '.pkl')
        f = open(root,"wb")
        pickle.dump(prototype_ci,f)
        f.close()
        np.save(os.path.join(proto_epoch_dir, 'vis_count' + str(epoch_number) + '.npy'),
                vis_count)

    end = time.time()
    print('\tvis time: \t{0}'.format(end -  start))
    print('total hit:',vis_count.sum())
    print('total prototype visualized', len(vis_count)-sum(vis_count ==0))
    print('mean patches hit',vis_count.mean())
    print('vis done')
    return vis_count