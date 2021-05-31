from .basic_net import Network
import numpy as np
import torch

genotype = [['11', '11',
                                      '11', '11',
                                      '11', '01',
                                      '11', '01',
                                      '10', '11',
                                      '01', '01',
                                      '01', '11']]
def robnet(genotype_list, **kwargs):
    return Network(genotype_list=genotype_list, **kwargs)

def count_parameters_in_MB(model):
    # import ipdb; ipdb.set_trace()
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if 'graph_bn' not in name and 'GCN' not in name and 'transformation' not in name) / 1e6

if __name__ == '__main__':
    model = robnet(genotype, C=64,
                   num_classes=10,
                   layers=33,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=True,
                   AdPoolSize=1)
    print(count_parameters_in_MB(model))
    