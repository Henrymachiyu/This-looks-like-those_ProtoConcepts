import time
import torch
import torch.nn.functional as F
import pdb
import math

from helpers import list_of_distances, make_one_hot
from settings import sep_cost_filter, sep_cost_cutoff, use_cap, debug, sub_mean, ltwo

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, clst_k=1,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(dataloader):
        #print('current index for loader is',i)
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req: #, torch.autograd.detect_anomaly():
            # nn.Module has implemented __call__() function
            # so no need to call .forward

            output, min_distances = model(input)
            cap = torch.linalg.norm(model.module.cap_width_l2)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            
            max_dist = (model.module.prototype_shape[1]
                        * model.module.prototype_shape[2]
                        * model.module.prototype_shape[3])

            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
 
            #calculate cluster cost
            if clst_k == 1: 
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            else: 
                # clst_k is a hyperparameter that lets the cluster cost apply in a "top-k" fashion: the original cluster cost is equivalent to the k = 1 casse
                inverted_distances, _ = torch.topk((max_dist - min_distances) * prototypes_of_correct_class, k = clst_k, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)
            #calculate separation cost
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg separation cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
            
            l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            foo = total_cross_entropy / n_batches
            if math.isnan(foo): 
                print('iteration: ', i)
                print('detected nan!')
                print('total cross entropy: ', total_cross_entropy)
                print('n_batches: ', n_batches)
                print('cross entropy for this batch: ', cross_entropy.item())
                print('target: ', target)
                print('output: ', output)
                torch.save(model, 'nanmodel.pth')
                exit()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                              + coefs['clst'] * cluster_cost
                              + coefs['sep'] * separation_cost
                              + coefs['l1'] * l1
                              +coefs['cap']*cap)
                    if math.isnan(loss): 
                        print('loss is nan!')
                        print('cross_entropy: ', cross_entropy)
                        print('cluster_cost: ', cluster_cost)
                        print('separation_cost: ', separation_cost)
                        print('l1: ', l1)
                        exit()
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1
                          )
                else:
                    loss = cross_entropy + 1e-4 * l1
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    log('\tcap loss: \t\t{0}'.format(cap))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, clst_k=1, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, clst_k = clst_k, coefs=coefs, log=log)

def test(model, dataloader, class_specific=False, clst_k=1, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, clst_k=clst_k, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')
        

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')

def stable_log(p): 
    one_mask = (p == 0.).nonzero()
    return torch.log(p + one_mask)