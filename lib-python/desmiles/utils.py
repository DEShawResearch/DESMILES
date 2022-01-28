import os
import numpy as np
from pathlib import Path
from .models import *
from .learner import *
from .data import DesmilesLoader, FpSmilesList
from .config import DATA_DIR
from rdkit import Chem
from rdkit.Chem import Draw,rdMolDescriptors,AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import scipy.sparse
from functools import partial
import torch
import torch.nn.functional as F
from fastai.text import Vocab
from fastai.basic_data import DataBunch


def load_pretrained_desmiles(path,
                             return_learner=False,
                             return_rnn=False,
                             fp_emb_sz=2000,
                             emb_sz=400,
                             nh=2000,
                             nl=5,
                             itos_fn=os.path.join(DATA_DIR, 'pretrained', 'id.dec8000'),
                             bs=200,
                             device=None,
                             with_opt=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(path)
    path = path.parent / path.stem
    itos = [s.strip() for i,s in enumerate(open(itos_fn, encoding='utf-8'))]
    vocab = Vocab(itos)

    trn_ds = FpSmilesList(np.asarray([]),np.asarray([]))
    val_ds = FpSmilesList(np.asarray([]),np.asarray([]))

    trn_dl = DesmilesLoader(trn_ds, bs=bs, vocab=vocab)
    val_dl = DesmilesLoader(val_ds, bs=bs, vocab=vocab)
    db = DataBunch(trn_dl, val_dl)
    learner = desmiles_model_learner(db, drop_mult=0.7, fp_emb_sz=fp_emb_sz, emb_sz=emb_sz, nh=nh, nl=nl, pad_token=0, bias=False)
    # purge makes loading really slow!!
    learner.load(path, purge=False, device=device, with_opt=with_opt)
    if return_learner:
        learner.model = learner.model.to(device)
        return learner
    if return_rnn:
        rnn_desmiles = RecurrentDESMILES(learner.model).eval()
        return rnn_desmiles.to(device)
    model = learner.model
    return model.to(device)


def load_old_pretrained_desmiles(path,
                                 return_learner=False,
                                 return_rnn=False,
                                 bs=200,
                                 fp_emb_sz=2000,
                                 emb_sz=400,
                                 nh=2000,
                                 nl=5,
                                 clip=0.3,
                                 alpha=2.,
                                 beta=1.,
                                 itos_fn=os.path.join(DATA_DIR, 'pretrained', 'id.dec8000'),
                                 device=None):
    '''
    Load a DESMILES model whose weights were generated in pytorch 0.4
    '''
    path = Path(path)
    path = path.parent / path.stem
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(itos_fn, encoding='utf-8') as itos_file:
        itos = [s.strip() for i,s in enumerate(itos_file)]
    vocab = Vocab(itos)
    bs=1
    trn_ds = FpSmilesList(np.asarray([]), np.asarray([]), vocab)
    val_ds = FpSmilesList(np.asarray([]), np.asarray([]), vocab)
    trn_dl = DesmilesLoader(trn_ds, vocab=vocab, bs=bs, sampler=None)
    val_dl = DesmilesLoader(val_ds, vocab=vocab, bs=bs, sampler=None)
    db = DataBunch(trn_dl, val_dl, path=".")
    learn = desmiles_model_learner(db, drop_mult=0.7, fp_emb_sz=fp_emb_sz, emb_sz=emb_sz, nh=nh, nl=nl, pad_token=0, bias=False, clip=clip, alpha=alpha, beta=beta)
    learn.model.reset()
    learn.load_old(path, strict=False)
    learn.model.eval()
    learn.model.reset()
    if return_learner:
        learn.model = learn.model.to(device)
        return learn
    if return_rnn:
        rnn_desmiles = RecurrentDESMILES(learn.model).eval()
        return rnn_desmiles.to(device)
    model = learn.model
    return model.to(device)


def smiles_to_fingerprint(smiles_str, sparse=False, as_tensor=False):
    rdmol = Chem.MolFromSmiles(smiles_str)
    fp = np.concatenate([np.asarray(GetMorganFingerprintAsBitVect(rdmol, 2, useChirality=True), dtype=np.int8), np.asarray(
        GetMorganFingerprintAsBitVect(rdmol, 3, useChirality=True), dtype=np.int8)])
    if sparse:
        return scipy.sparse.csr_matrix(fp)
    if as_tensor:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(fp.astype(np.float32)).to(device)
    return fp


def process_smiles(sm):
    m = Chem.MolFromSmiles(sm)
    AllChem.Compute2DCoords(m)
    return m


def image_of_mols(smiles_list, molsPerRow=5, subImgSize=(200,200), labels=None):
    mols = [process_smiles(sm) for sm in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=subImgSize, useSVG=True, legends=labels)
    return img

def accuracy4(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes, excluding tokens 0--3."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input[targs > 3] == targs[targs > 3]).float().mean()

def decoder(idx_vec, itos):
    """Return a SMILES string from an index vector (deals with reversal)"""
    if len(idx_vec) < 2:
        return ""
    if idx_vec[1] == 1:  # SMILES string is in fwd direction
        return ''.join(itos[x] for x in idx_vec if x > 3)
    if idx_vec[1] == 2:  # SMILES string is in bwd direction
        return ''.join(itos[x] for x in idx_vec[::-1] if x > 3)
    else: # don't know how to deal with it---do your best
        return ''.join(itos[x] for x in idx_vec if x > 3)

def get_default_decoder():
    from functools import partial
    itos_fn=os.path.join(DATA_DIR, 'pretrained', 'id.dec8000')
    with open(itos_fn, encoding='utf-8') as itos_file:
        itos = [s.strip() for i,s in enumerate(itos_file)]
    return partial(decoder, itos=itos)

default_decoder = get_default_decoder()
def smiles_idx_to_string(smiles_idx, decoder=default_decoder):
    return decoder(smiles_idx[smiles_idx > 0].tolist())
