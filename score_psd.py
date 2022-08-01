#!/usr/bin/env python3
#Author: Jared Adolf-Bryfogle
#Score proteins using PSD.

from argparse import ArgumentParser
import sys

from seq_des.common.load import *
from seq_des.scoring import get_log_p
from seq_des.util import sampler_util
from collections import defaultdict
import seq_des.common.atoms as atoms
import pandas



if __name__ == "__main__":
    parser = ArgumentParser("This script scores individual proteins or lists of proteins and outputs the data into an "
                            "output CSV file for further analysis. "
                            "Setup: You will need both PATH and PYTHONPATH to include the PSD root.")


    parser.add_argument("--pdblist", '-l',
                        help = "A list of paths to PDBs for scoring.")

    parser.add_argument("--pdb", '-s',
                        help = "A path to a PDB file for scoring")


    parser.add_argument('--out_dir', '-d',
                        default = "psd_calcs",
                        help = "Root output directory.")

    options = parser.parse_args()

    if not options.pdblist and not options.pdb:
        sys.exit("A pdblist or pdb file must be passed to the script.")




    #Make needed directories
    if not os.path.exists(options.out_dir):
        os.mkdir(options.out_dir)

    log_dir = options.out_dir+'/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


    pdbs = []
    if options.pdb:
        pdbs.append(options.pdb)
    else:
        for line in open(options.pdblist, 'r'):
            line = line.strip()
            if not line: continue
            if line.startswith('#'): continue

            pdbs.append(line)

    #Load models
    classifiers = load_models(use_cuda=torch.cuda.is_available(), nic=len(atoms.atoms) + 1 + 21)
    for classifier in classifiers:
        classifier.eval()

    #Run calculations, append the data and create a DF for csv output
    all_data = []
    for pdb in pdbs:
        data = defaultdict()
        p_mean, p_all = get_log_p(pdb, log_dir)

        base = ".".join(os.path.basename(pdb).split('.')[0:-1])
        print(base)
        data['name'] = base
        data['decoy'] = pdb
        data['log_p_mean'] = p_mean
        print(base, p_mean)

        #Concatonate individual scores for further downstream parsing.
        out = []
        for i in range(len(p_all)):
            res = i+1
            out.append(f'{res}={p_all[res]:.3f}')

        data['log_p_res'] = "|".join(out)
        all_data.append(data)

    df = pandas.DataFrame.from_records(all_data)
    df.to_csv(options.out_dir+'/psd_scores.csv')
