import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from util.receptive_field import compute_proto_layer_rf_info_v2

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

class TESNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes,
                 cap_width, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(TESNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        ####### cap unit
        self.cap_activation = nn.Softplus()
        self.cap_width_l2 = nn.Parameter(cap_width * torch.ones(1, self.num_prototypes), 
                                        requires_grad=True)
        self.max_act_l2 = torch.log(torch.tensor(1)/self.epsilon)
        # prune like protopnet doesn't work as the subspace sep and ortho loss would have diff in dimension
        self.non_vis_mask = nn.Parameter(torch.ones(self.num_prototypes),
                                 requires_grad=False)
        self.pruned = False
        ######################
        self.prototype_activation_function = prototype_activation_function #log

        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features #

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

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
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):

        x = self.features(x)
        x = self.add_on_layers(x)

        return x
    
    def soft_clamp(self, x, val=1.0, upper=True): 
        if upper: 
            return torch.min(x, val + x - x.detach())
        else: 
            return torch.max(x, val + x - x.detach())
        
    def _cosine_convolution(self, x):

        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors,p=2,dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        distances = -distances

        return distances
    
    def _project2basis(self,x):

        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        return distances

    def prototype_distances(self, x):

        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)

        return project_distances,cosine_distances

    def distance_2_similarity(self, distances):

        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def global_min_pooling(self,distances):

        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        return min_distances

    def global_max_pooling(self,distances):

        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances
    
    def prune(self, stats_arr):
        non_vis_proto = np.argwhere(stats_arr == 0).flatten() # [1, m]
        self.non_vis_mask[non_vis_proto] = 0.0
        self.pruned = True
        assert(torch.all(self.non_vis_mask[non_vis_proto] == 0.0))

    def forward(self, x, vis_cap = False):
        
        if not vis_cap:
            project_distances,cosine_distances = self.prototype_distances(x)
            cosine_min_distances = self.global_min_pooling(cosine_distances)
            project_max_distances = self.global_max_pooling(project_distances)
            prototype_activations = project_max_distances
            cap_factor = self.cap_activation(self.cap_width_l2) # [1, m]
            protoact_capped = self.soft_clamp(prototype_activations + cap_factor, val= self.max_act_l2, 
                    upper = True) - cap_factor.detach()
            if self.pruned:
                # basically set non-vis prototype activation to 0 so that it won't contribute to any of the
                # computation. 
                protoact_capped = self.non_vis_mask*protoact_capped.clone() # [bsz, m]
            logits = self.last_layer(protoact_capped)
            return logits, cosine_min_distances,protoact_capped, cap_factor
        elif vis_cap:
            big_proto_act,_ = self.prototype_distances(x)
            cap_factor = self.cap_activation(self.cap_width_l2).unsqueeze(2).unsqueeze(3)
            return big_proto_act, cap_factor


    def push_forward(self, x):

        conv_output = self.conv_features(x) #[batchsize,128,14,14]
        
        distances = self._project2basis(conv_output)
        #distances = - distances
        cap_factor = self.cap_activation(self.cap_width_l2).unsqueeze(2).unsqueeze(3)
        return conv_output, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

def construct_TesNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 128, 1, 1), num_classes=200,
                    cap_width = 6.0,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,#224
                                                         layer_filter_sizes=layer_filter_sizes,#
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return TESNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 cap_width = cap_width,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

