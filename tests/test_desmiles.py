'''
Documentation for desmiles

'''
from __future__ import print_function
import os
import pytest
from pytest import approx
import subprocess
from desmiles.utils import load_old_pretrained_desmiles
from desmiles.config import DATA_DIR
from pathlib import Path
from functools import partial
import torch
from desmiles.models import *
from rdkit import Chem
import numpy as np


REGRESSION_DIR = Path(os.path.join(DATA_DIR, 'regression'))
PRETRAINED_DIR = Path(os.path.join(DATA_DIR, 'pretrained'))

def get_recovered_smiles(num_smiles=10):
    return [s.strip() for s in open(REGRESSION_DIR / 'recovered_2000_400_2000_5.smi')][:num_smiles]

def get_not_recovered_smiles(num_smiles=10):
    return [s.strip() for s in open(REGRESSION_DIR / 'not_recovered_2000_400_2000_5.smi')][:num_smiles]

def get_model_fn():
    return PRETRAINED_DIR / 'model_2000_400_2000_5'

def get_old_astar_results():
    "Return a dictionary of SMILES to top 100 results from old astar implementation of old model"
    import json
    with open(os.path.join(REGRESSION_DIR, 'astar_data.json'), 'r') as fp:
        results = json.load(fp)
    return results


def test_desmiles_imports():
    import torch
    import desmiles
    import rdkit
    print("It passes!")


def test_load_old_pretrained_model():
    print("Loading pretrained model")
    model_fn = get_model_fn()
    model = load_old_pretrained_desmiles(model_fn)
    assert not model.training

def test_load_old_pretrained_learner():
    import fastai
    print("Loading pretrained model")
    model_fn = get_model_fn()
    learner = load_old_pretrained_desmiles(model_fn, return_learner=True)
    assert any([f.func == fastai.train.GradientClipping for f in learner.callback_fns])
    assert not learner.model.training
    assert len(learner.callbacks) == 1
    assert type(learner.callbacks[0]) is fastai.callbacks.rnn.RNNTrainer


def get_pretrained_rnn():
    model_fn = get_model_fn()
    model = load_old_pretrained_desmiles(model_fn, return_rnn=True)
    return model


def get_pretrained_model():
    model_fn = get_model_fn()
    model = load_old_pretrained_desmiles(model_fn)
    return model


@pytest.fixture(scope="module")
def pretrained_desmiles_rnn():
    model = get_pretrained_rnn().to("cpu")
    return model


@pytest.fixture(scope="module")
def pretrained_desmiles_model():
    model = get_pretrained_model().to("cpu")
    return model


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


def get_random_input():
    device = "cuda" if torch.cuda.is_available() else "cpu"   
    input_seq = torch.zeros(300,26, dtype=torch.int64) # create a batch of size 300
    input_seq[:,0] = 3 # first token is 3
    input_seq[:,1] = torch.randint(1,3,(300,)) # second token is random fwd/bwd
    max_length = 26 # max length of any sequence is 26. sequences are ordered by length
    for i in range(300): # for every elemeent in the batch
        max_length = torch.randint(5,max_length+1,(1,)).item() # sample a new length (pad the rest)
        input_seq[i,2:max_length] = torch.randint(1,8000,(max_length - 2,)) # randomly sample tokens for the sequence
    lengths = (input_seq > 0).sum(dim=1) # get the lengths
    fps = torch.randint(1, (300,4096), device=device).type(torch.float) # get random fingerprints
    input_seq = input_seq.to(device)
    return input_seq, fps, lengths


itos_fn=os.path.join(PRETRAINED_DIR, 'id.dec8000')
itos = [s.strip() for i,s in enumerate(open(itos_fn, encoding='utf-8'))]
decoder = partial(decoder, itos=itos)
isLeafNode = lambda s_vec: (s_vec[-1] == 2 or s_vec[-1] == 1 or len(s_vec) == 26) if len(s_vec) > 3 else False


def test_output_old_pytorch():
    """This test compares a set of random inputs/outputs
    from the early version of DESMILES against those of the later versions.
    The early version of DESMILES used pytorch 0.4 and fastai 0.0.2,
    and the local version of the test creates a random input every time
    with explicit dependencies for running the original code.
    To reduce the burden on keeping multiple installations of the dependencies
    we've only kept a number of specific inputs and outputs and read them from torch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    datasets = [
        [ os.path.join(REGRESSION_DIR, 'tmpo2o5u_i0.pt'),
          os.path.join(REGRESSION_DIR, 'tmp5g4_g26n.pt'),
          os.path.join(REGRESSION_DIR, 'tmpas2eooku.pt') ],
        [ os.path.join(REGRESSION_DIR, 'tmp4qe8p4rd.pt'),
          os.path.join(REGRESSION_DIR, 'tmp479_2pzi.pt'),
          os.path.join(REGRESSION_DIR, 'tmp90vhskjg.pt') ]
    ]
    model = get_pretrained_model()  # load 3.7 pytorch 1.0+ desmiles from a dump of the old model
    for input_seq_fn, input_fp_fn, output_fn in datasets:
        input_seq = torch.load(input_seq_fn, map_location=device)
        input_fp = torch.load(input_fp_fn, map_location=device)
        lengths = (input_seq > 0).sum(dim=1) # get the lengths
        with torch.no_grad():
            outputs = model(input_seq.transpose(0,1), input_fp, lengths) # get output of desmiles
        outputs_torch_04 = torch.load(output_fn)  # load desmiles 0.4 output
        assert(torch.abs(outputs_torch_04[0] - outputs[0].to("cpu")).max() < 2e-2)
        for o_torch_04, o in zip(outputs_torch_04[1], outputs[1]):
            assert(torch.abs(o_torch_04 - o.to('cpu')).max() < 2e-2)
        for o_torch_04, o in zip(outputs_torch_04[2], outputs[2]):
            assert(torch.abs(o_torch_04 - o.to('cpu')).max() < 2e-2)
    return


def test_split_desmiles_first_layer():
    "Test splitting of DESMILES at 1st layer, as per paper"
    model = get_pretrained_model()
    fp_to_embedding = get_fp_to_embedding_model(model, first_layer=True)
    embedded_fp_to_smiles = get_embedded_fp_to_smiles_model(model, first_layer=True)
    input_seq, fps, lengths = get_random_input()
    with torch.no_grad():
        output = model(input_seq.transpose(0,1), fps, lengths)
        embedding = fp_to_embedding(fps)
        output_2 = embedded_fp_to_smiles(input_seq.transpose(0,1), embedding, lengths)
    assert(torch.abs(output[0] - output_2[0]).max() < 1e-4)
    for o, o2 in zip(output[1], output_2[1]):
        assert(torch.abs(o - o2).max() < 1e-4)
    for o, o2 in zip(output[2], output_2[2]):
        assert(torch.abs(o - o2).max() < 1e-4)


def test_split_desmiles_second_layer():
    "Test splitting of DESMILES at 2nd layer, just in case"
    model = get_pretrained_model()
    fp_to_embedding = get_fp_to_embedding_model(model, first_layer=False)
    embedded_fp_to_smiles = get_embedded_fp_to_smiles_model(model, first_layer=False)
    input_seq, fps, lengths = get_random_input()
    with torch.no_grad():
        output = model(input_seq.transpose(0,1), fps, lengths)
        embedding = fp_to_embedding(fps)
        output_2 = embedded_fp_to_smiles(input_seq.transpose(0,1), embedding, lengths)
    assert(torch.abs(output[0] - output_2[0]).max() < 1e-4)
    for o, o2 in zip(output[1], output_2[1]):
        assert(torch.abs(o - o2).max() < 1e-4)
    for o, o2 in zip(output[2], output_2[2]):
        assert(torch.abs(o - o2).max() < 1e-4)


def test_recurrent_desmiles():
    "Test recurrent version of DESMILES used in beam and astar search."
    model = get_pretrained_model()
    input_seq, fps, lengths = get_random_input()
    with torch.no_grad():
        output = model(input_seq.transpose(0,1), fps, lengths)
        rdesmiles = RecurrentDESMILES(model)
        rdesmiles.embed_fingerprints(fps)
        output_2 = rdesmiles(input_seq.transpose(0,1))
        # Output of sequence-input recurrent astar model is BS x SL x vocab_size (8000).
        # Output of fp-sequence-input model is -1 (SLxBS) x vocab_size (8000).
    assert(torch.abs(output_2.transpose(0,1).reshape(-1, 8000) - output[0]).max() < 1e-5)

def test_recurrent_desmiles_from_embedding_first_layer():
    "Test recurrent version of DESMILES used in beam and astar search."
    model = get_pretrained_model()
    fp_to_embedding = get_fp_to_embedding_model(model, first_layer=True)
    # use only the second linear layer to precalculate fingerprint embedding
    rdesmiles = RecurrentDESMILES(model, fp_embedding_layers=(1,))
    input_seq, fps, lengths = get_random_input()
    with torch.no_grad():
        output = model(input_seq.transpose(0,1), fps, lengths)
        embedding = fp_to_embedding(fps)
        rdesmiles.embed_fingerprints(embedding)
        output_2 = rdesmiles(input_seq.transpose(0,1))
        # Output of sequence-input recurrent astar model is BS x SL x vocab_size (8000).
        # Output of fp-sequence-input model is -1 (SLxBS) x vocab_size (8000).
    assert(torch.abs(output_2.transpose(0,1).reshape(-1, 8000) - output[0]).max() < 1e-5)

def test_recurrent_desmiles_from_embedding_second_layer():
    "Test recurrent version of DESMILES used in beam and astar search."
    model = get_pretrained_model()
    fp_to_embedding = get_fp_to_embedding_model(model, first_layer=False)
    # the fingerprint is already fully embedded so fp_embedding_layers is empty
    rdesmiles = RecurrentDESMILES(model, fp_embedding_layers=())
    input_seq, fps, lengths = get_random_input()
    with torch.no_grad():
        output = model(input_seq.transpose(0,1), fps, lengths)
        embedding = fp_to_embedding(fps)
        rdesmiles.embed_fingerprints(embedding)
        output_2 = rdesmiles(input_seq.transpose(0,1))
        # Output of sequence-input recurrent astar model is BS x SL x vocab_size (8000).
        # Output of fp-sequence-input model is -1 (SLxBS) x vocab_size (8000).
    assert(torch.abs(output_2.transpose(0,1).reshape(-1, 8000) - output[0]).max() < 1e-5)


def test_against_sequential_astar_mem_safe():
    from desmiles.utils import smiles_to_fingerprint
    import desmiles.decoding
    from desmiles.decoding.astar import AstarTreeParallel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old_astar_results = get_old_astar_results()
    smiles_to_test = old_astar_results.keys()
    for smiles in smiles_to_test:
        print(f"Testing A* with smiles: {smiles}")
        # get fingerprint: output of smiles_to_fingerprint is fp_size (4096),
        # which is reshaped to BS x fp_size,
        # which is the input to the sequence-input (recurrent) DESMILES
        fp = torch.tensor(smiles_to_fingerprint(smiles).astype(float).reshape(1,-1), dtype=torch.float, device=device)
        sequential_results = old_astar_results[smiles]
        parallel_results = []
        model = get_pretrained_rnn()
        astar_par = AstarTreeParallel(fp, model, num_expand=100)
        parallel_results = []
        for _ in range(len(sequential_results)):
            score, smiles_vector = next(astar_par)
            # The new code keeps the padding in the output, removed here:
            smiles_vector = [x for x in smiles_vector.tolist() if x > 0]
            smiles_string = decoder(smiles_vector)
            parallel_results.append([score, smiles_string, smiles_vector])
        # Numerical accuracy is such that sometimes the order is swapped.
        # So this check tests one before and one after.
        # only check input molecule up to the penultimate SMILES.
        for i, (pr, sr) in enumerate(zip(parallel_results[:-1], sequential_results[:-1])):
            assert ((pr[0] == approx(sr[0], rel=1e-2)) or
                    (pr[0] == approx(sequential_results[i+1][0], rel=1e-2)) or
                    (pr[0] == approx(sequential_results[i-1][0], rel=1e-2)))
            assert (pr[1:] == sr[1:]) or (pr[1:] == sequential_results[i-1][1:]) or ((pr[1:] == sequential_results[i+1][1:]))
    return


def test_against_sequential_astar_not_mem_safe():
    from desmiles.utils import smiles_to_fingerprint
    import desmiles.decoding
    from desmiles.decoding.astar import AstarTreeParallelNotSafe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old_astar_results = get_old_astar_results()
    smiles_to_test = old_astar_results.keys()
    for smiles in smiles_to_test:
        print(f"Testing A* with smiles: {smiles}")
        # get fingerprint: output of smiles_to_fingerprint is fp_size (4096),
        # which is reshaped to BS x fp_size,
        # which is the input to the sequence-input (recurrent) DESMILES
        fp = torch.tensor(smiles_to_fingerprint(smiles).astype(float).reshape(1,-1), dtype=torch.float, device=device)
        sequential_results = old_astar_results[smiles]
        parallel_results = []
        model = get_pretrained_rnn()
        astar_par = AstarTreeParallelNotSafe(fp, model, num_expand=100)
        parallel_results = []
        for _ in range(len(sequential_results)):
            score, smiles_vector = next(astar_par)
            # The new code keeps the padding in the output, removed here:
            smiles_vector = [x for x in smiles_vector.tolist() if x > 0]
            smiles_string = decoder(smiles_vector)
            parallel_results.append([score, smiles_string, smiles_vector])
        # Numerical accuracy is such that sometimes the order is swapped.
        # So this check tests one before and one after.
        # only check input molecule up to the penultimate SMILES.
        for i, (pr, sr) in enumerate(zip(parallel_results[:-1], sequential_results[:-1])):
            assert ((pr[0] == approx(sr[0], rel=1e-2)) or
                    (pr[0] == approx(sequential_results[i+1][0], rel=1e-2)) or
                    (pr[0] == approx(sequential_results[i-1][0], rel=1e-2)))
            assert (pr[1:] == sr[1:]) or (pr[1:] == sequential_results[i-1][1:]) or ((pr[1:] == sequential_results[i+1][1:]))
    return


def test_rdkit_molecule_chirality():
    s1 = "C[C@@H]1CCCNC1[C@@H](C)N"
    s2 = "C[C@@H](N)C1NCCC[C@H]1C"
    mol = Chem.MolFromSmiles(s1)
    assert Chem.CanonSmiles(s1) == s2
    assert Chem.MolToSmiles(mol) == s2


def test_one_cycle_learner():
    from desmiles.learner import OriginalFastaiOneCycleScheduler, Learner
    from fastai.basic_data import DataBunch
    n=1000
    sigma=0.1
    x = np.linspace(-1,1, n) + (np.random.randn(n) * sigma)
    y = x**2
    x_t = torch.tensor(x, dtype=torch.float).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float).unsqueeze(1)

    trn_ds = torch.utils.data.TensorDataset(x_t, y_t)
    val_ds = torch.utils.data.TensorDataset(x_t, y_t)
    trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=10, shuffle=False)
    db = DataBunch(trn_loader, val_loader)
    model = torch.nn.Sequential(torch.nn.Linear(1,100), torch.nn.ReLU(), torch.nn.Linear(100,1), torch.nn.ReLU())
    learner = Learner(db, model, loss_func=torch.nn.functional.mse_loss)
    div_factor=2
    one_cycle_linear_cb = OriginalFastaiOneCycleScheduler(learner, 0.002, div_factor=div_factor)
    learner.fit(1, callbacks=[one_cycle_linear_cb])
    assert max(learner.recorder.lrs) / learner.recorder.lrs[0] == div_factor
