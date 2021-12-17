import torch.nn as nn
import torch
import seq_des.models as models
import seq_des.common.atoms as atoms
from .paths import *

def load_model(model, use_cuda=True, nic=len(atoms.atoms)):
    print("Loading model")
    if os.path.exists(get_project_root()+'/'+model):
        model = get_project_root()+'/'+model

    classifier = models.seqPred(nic=nic)
    if use_cuda:
        classifier.cuda()
    if use_cuda:
        state = torch.load(model)
    else:
        state = torch.load(model, map_location="cpu")
    for k in state.keys():
        if "module" in k:
            print("MODULE")
            classifier=classifier
        break
    if use_cuda:
        classifier.load_state_dict(torch.load(model))
    else:
        classifier.load_state_dict(torch.load(model, map_location="cpu"))
    return classifier


def load_models(model_list = None, use_cuda=True, nic=len(atoms.atoms)):
    """
    If model list is none, load default models.  Also looks in seq_des root!
    :param model_list:
    :param use_cuda:
    :param nic:
    :return:
    """
    if not model_list:
        model_list = [
            "models/conditional_model_0.pt",
            "models/conditional_model_1.pt",
            "models/conditional_model_2.pt",
            "models/conditional_model_3.pt",
        ]

    classifiers = []
    for model in model_list:
        print(get_project_root()+'/'+model)
        if os.path.exists(get_project_root()+'/'+model):
            model = get_project_root()+'/'+model
        classifier = load_model(model, use_cuda=use_cuda, nic=nic)
        classifiers.append(classifier)
    return classifiers
