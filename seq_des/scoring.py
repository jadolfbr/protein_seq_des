
import torch
import os
from seq_des.common.load import *
from seq_des.util import sampler_util
from collections import defaultdict
import seq_des.common.atoms as atoms



def get_log_p_from_seq_des(pose, logdir, classifiers=None):
    """
    Returns the logp_mean and logp per residue from seq des inference code.
    Pass models in to speed up inference speed
    :param logdir:
    :return:
    """

    if not os.path.exists(logdir): os.mkdir(logdir)

    if not classifiers:
        classifiers = load_models(use_cuda=torch.cuda.is_available(), nic=len(atoms.atoms) + 1 + 21)
        for classifier in classifiers:
            classifier.eval()

    (res_label, log_p_per_res, log_p_mean, logits, chi_feat, chi_angles,
     chi_mask,) = sampler_util.get_energy(
        classifiers, pose, log_path=logdir, include_rotamer_probs=1, use_cuda=torch.cuda.is_available(),
    )

    p_mean = log_p_mean.item()

    out = defaultdict()
    for i in range(0, len(log_p_per_res)):
        out[i+1] = log_p_per_res[i].item()

    return p_mean, out