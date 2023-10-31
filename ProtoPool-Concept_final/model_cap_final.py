from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax


from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
import numpy as np

from utils import compute_proto_layer_rf_info_v2

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class Protopool_cap(nn.Module):
    def __init__(self, num_prototypes: int, num_descriptive: int, num_classes: int,
                 use_thresh: bool = False, arch: str = 'resnet34', pretrained: bool = True,
                 add_on_layers_type: str = 'linear', prototype_activation_function: str = 'log',
                 proto_depth: int = 128, use_last_layer: bool = False, inat: bool = False,
                 cap_width:int = 1, epsilon:float = 1e-4):
        super().__init__()
        self.num_classes = num_classes #n
        self.epsilon = epsilon
        self.num_descriptive = num_descriptive #k
        self.num_prototypes = num_prototypes #m
        self.proto_depth = proto_depth # D
        self.prototype_shape = (self.num_prototypes, self.proto_depth, 1, 1) #202 * 256 
        self.use_thresh = use_thresh # T
        self.arch = arch #T
        self.pretrained = pretrained # resnet 50
        self.inat = inat # T
        self.pruned = False
        if self.use_thresh:
            self.alfa = nn.Parameter(torch.Tensor(1, num_classes, num_descriptive))
            nn.init.xavier_normal_(self.alfa, gain=1.0)
        else:
            self.alfa = 1
            self.beta = 0

        ###### prototypes ####################
        # prototype shape -- > num_proto 202 , depth 256 , 1, 1 (patches)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        self.prototype_activation_function = prototype_activation_function # log 

        #### slots initialization ###########
        # Class Num -> n, num_prototypes -> m, num_descriptive --> k (along with notation in paper)
        self.proto_presence = torch.zeros(num_classes, num_prototypes, 
                                        num_descriptive) # [n, m, k]
        self.proto_presence = nn.Parameter(self.proto_presence, requires_grad=True)# [n, m, k]
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)

        #### CAP ################################################

        self.max_act_l2 = torch.log(torch.tensor(1)/self.epsilon)
        self.cap_width_l2 = nn.Parameter(cap_width * torch.ones(1, self.num_prototypes), 
                                        requires_grad=True)
        self.cap_activation = nn.Softplus()# soft plus 
    
        self.non_vis_mask = nn.Parameter(torch.ones(self.proto_presence.shape),
                                 requires_grad=False)
        ############ Base arch preprocess ##########################
        if self.inat:# only applies to resnet 50 
            self.features = base_architecture_to_features['resnet50'](pretrained=pretrained, inat=True)
        else:
            self.features = base_architecture_to_features[self.arch](pretrained=pretrained)
        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base architecture type NOT implemented')

        ################### add on layers ##########
        if add_on_layers_type == 'bottleneck':

            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)

        elif add_on_layers_type == 'regular':
            add_on_layers = [
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, 
                          out_channels=self.prototype_shape[1],
                          kernel_size=1),
                nn.Sigmoid()
            ]
            self.add_on_layers = nn.Sequential(*add_on_layers)

        elif add_on_layers_type == 'nosig':     
            self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, 
            out_channels=self.prototype_shape[1], 
            kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.prototype_shape[1], 
            out_channels=self.prototype_shape[1], 
            kernel_size=1)
            )

        elif add_on_layers_type == 'reg_nosig':
            self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, 
            out_channels=self.prototype_shape[1], 
            kernel_size=1))
        
        else:
            raise NotImplementedError

        for layer in self.add_on_layers.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        ########## last layer ###############
    
        self.use_last_layer = use_last_layer
        if self.use_last_layer:

            self.prototype_class_identity = torch.zeros(self.num_descriptive * self.num_classes, 
            self.num_classes) # kn, n
            for j in range(self.num_descriptive * self.num_classes):
                self.prototype_class_identity[j, j // self.num_descriptive] = 1

            self.last_layer = nn.Linear(self.num_descriptive * self.num_classes, 
                                        self.num_classes, 
                                        bias=False) # k*n --> n 
            positive_one_weights_locations = torch.t(self.prototype_class_identity) # n k*n
            negative_one_weights_locations = 1 - positive_one_weights_locations

            correct_class_connection = 1
            incorrect_class_connection = 0 # -0.5

            #set weight for correct class connection as 1, 0 o.w.
            self.last_layer.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)
        else:
            self.last_layer = nn.Identity()

    ############# util functions ############################
    def softplus_inv(self, x):
        result = torch.log(torch.exp(torch.tensor(x))-1).item()
        return result
        
    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        return l2 norm^2  of f(x) and pj
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances
    
    def prototype_distances(self, x):
        '''
        x is the raw input
        calculate l2 norm squared with raw input 
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)  # [b, m, h, w]
        return distances  # [b, k, h, w], [b, m, h, w]?

    def distance_2_similarity(self, distances):  # [b,c,n]
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))

        elif self.prototype_activation_function == 'linear':
            if self.use_thresh:
                distances = distances  # * torch.exp(self.alfa)  # [b, c, n]
            return 1 / (distances + 1)
        else:
            raise NotImplementedError

    def _mix_l2_convolution(self, distances, proto_presence):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        # distances [b, p]
        # proto_presence [c, p, n]
        mixed_distances = torch.einsum('bp,cpn->bcn', distances, proto_presence)
        return mixed_distances  # [b, c, n]

    def get_map_class_to_prototypes(self):
        pp = gumbel_softmax(self.proto_presence * 10e3, dim=1, tau=0.5).detach()
        return np.argmax(pp.cpu().numpy(), axis=1)

    def soft_clamp(self, x, val=1.0, upper=True): 
        if upper: 
            return torch.min(x, val + x - x.detach())
        else: 
            return torch.max(x, val + x - x.detach())
    def prune(self, stats_arr):
        non_vis_proto = np.argwhere(stats_arr == 0).flatten() # the nonvis prototype idx 
        new_mask = torch.ones(self.proto_presence.shape).cuda()
        new_mask[:,non_vis_proto]  = 0.0
        self.non_vis_mask = nn.Parameter(new_mask,
                                 requires_grad=False)
        self.pruned = True
        # sanity check 
        assert(torch.all(self.non_vis_mask[:,non_vis_proto] == 0.0))

    ############### forward ################################
    def forward(self, x: torch.Tensor, gumbel_scale: int = 0,cap_all=False, vis_cap = False) -> \
            Tuple[torch.Tensor, torch.LongTensor]:
        ## q estimattion
        if gumbel_scale == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            # estimation of q for slots
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scale, dim=1, tau=0.5)
        # to speed up trainning slightly
        if self.pruned:
            proto_presence = self.non_vis_mask*proto_presence.clone()

        if not vis_cap: # for non-visualization computations
            ##### l2 norm calculation and related
            # bsz x m x h x w 
            distances = self.prototype_distances(x)  # [b, n, H, W] -> [b, m, h, w]
            '''
            we cannot refactor the lines below for similarity scores
            because we need to return min_distances
            '''
            # global min pooling
            min_distances = -F.max_pool2d(-distances,
                        kernel_size=(distances.size()[2],
                                    distances.size()[3])).squeeze()  # [b, m]
            cap_factor = self.cap_activation(self.cap_width_l2)# [1, m]
            avg_dist = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                            distances.size()[3])).squeeze()  # [b, m]
            proto_act = self.distance_2_similarity(min_distances)# [b, m]
            avg_act = self.distance_2_similarity(avg_dist)
            if len(avg_act.shape) == 1: # to accmodate only 1 input 
                avg_act = avg_act.unsqueeze(0)
            # choose if cap on the focal similarity
            if not cap_all:
                proto_act_capped = self.soft_clamp(proto_act.clone()+cap_factor, val= self.max_act_l2, 
                upper = True) - cap_factor.detach() # [b, m]
                comb_dist = self._mix_l2_convolution(proto_act_capped, proto_presence)  # [b, n, k]
                comb_avg = self._mix_l2_convolution(avg_act, proto_presence)  # [b, n, k]
                score = comb_dist- comb_avg # focal similarity

            elif cap_all:
                proto_act_focal = proto_act - avg_act 
                proto_act_capped = self.soft_clamp(proto_act_focal.clone()+cap_factor, val= self.max_act_l2, 
                upper = True) - cap_factor.detach() # [b, m]
                score = self._mix_l2_convolution(proto_act_capped, proto_presence) 

            if self.use_last_layer:
                prob = self.last_layer(score.flatten(start_dim=1))
            else:
                prob = score.sum(dim=-1)
            return prob, min_distances, proto_presence, cap_factor, avg_act, proto_act_capped  # [b,n] [b,m] [n, m, k]

        elif vis_cap: # for visualization
            distances = self.prototype_distances(x)
            cap_factor = self.cap_activation(self.cap_width_l2).unsqueeze(2).unsqueeze(3)
            big_proto_act = self.distance_2_similarity(distances)
            avg_d = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                                    distances.size()[3]))
            avg_act = self.distance_2_similarity(avg_d)
            if cap_all: # cap on the avg act and proto act
                big_proto_act_focal = big_proto_act - avg_act 
                big_proto_act_capped = self.soft_clamp(big_proto_act_focal+cap_factor, val= self.max_act_l2, 
                    upper = True) - cap_factor.detach()
                return cap_factor,avg_act, big_proto_act_focal
            else:
                big_proto_act_capped = self.soft_clamp(big_proto_act.clone() + cap_factor, val=self.max_act_l2, upper=True) \
                                                - cap_factor.detach()
            return cap_factor,avg_act, big_proto_act
                
        

    def __repr__(self):
        res = super(Protopool_cap, self).__repr__()
        return res

    def fine_tune_last_only(self):
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.add_on_layers.parameters():
            p.requires_grad = False
        self.prototype_vectors.requires_grad = False
        self.proto_presence.requires_grad = False
        for p in self.last_layer.parameters():
            p.requires_grad = True


        



        
