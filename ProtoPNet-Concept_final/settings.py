base_architecture = 'resnet152'
img_size = 224

num_classes = 200#196#200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
temp = 5.0

if add_on_layers_type == 'none': 
    prototype_shape = (2000, 512, 1, 1)
    #prototype_shape = (1960, 512, 1, 1)
    
else: 
    prototype_shape = (2000, 128, 1, 1)
    #prototype_shape = (1960, 128, 1, 1)

last_layer_type='single'
debug=False
hard_sparse=False
use_cap=True
ctrl=True
sub_mean=False
ltwo=True

cap_width = 6.0#7.5
clst_k = 10 

sep_cost_filter = 'cutoff' 
#none: regular separation cost
#cutoff: don't consider separation cost to classes where a prototype has weight > cutoff
sep_cost_cutoff = -0.05

spstr = 'hardsparse' if hard_sparse else 'softsparse'
capstr = 'cap' if use_cap else 'nocap'
lstr = 'l2' if ltwo else 'hs'

data_path =  ''# where you save the data
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
if debug: 
    train_batch_size = 5
    test_batch_size = 5
    train_push_batch_size = 5
else:
    train_batch_size = 50
    test_batch_size = 80
    train_push_batch_size = 50

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3, 
                       'last_layer': 1e-4}
joint_lr_step_size = 5

#experiment_run = experiment_run + 'll' + str(joint_optimizer_lrs['last_layer'])

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3,
                      'cap_width': 5e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'relu': 1e-3,
    'dist': 1.0,
    'cap': 0.01,
}

# name for experiment folder
if not ctrl: 
    experiment_run = 'dyn_ll' + capstr
elif sub_mean: 
    experiment_run = 'submean001'
else: 
    experiment_run = 'arr'

num_train_epochs = 15
num_warm_epochs = 5

if use_cap: 
    push_start = float('inf')
    push_epochs = []
else: 
    push_start = 10
    push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]