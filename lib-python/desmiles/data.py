from fastai.basics import *
from fastai.text import Vocab
import scipy.sparse
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw,rdMolDescriptors,AllChem


class FpSmiles(ItemBase):
    "Base item type in the fastai library."
    def __init__(self, ids,text,fp): 
        self.data=(ids, fp)
        self.text = text
    def __repr__(self): return str(self.text)
    def show(self): return imageOfMols([self.text], molsPerRow=1)
    def apply_tfms(self, tfms:Collection, **kwargs):
        if tfms: raise Exception('Not implemented')
        return self

class FpSmilesList(ItemList):
    "Basic `ItemList` for FpSmiles data."
    _bunch = DataBunch
    _label_cls = EmptyLabel

    def __init__(self, ids:NPArrayList, fps:'Scipy.Sparse', vocab:Vocab=None, pad_idx:int=0, **kwargs):
        super().__init__(ids, **kwargs)
        self.ids = ids
        self.fps = fps
        self.vocab,self.pad_idx = vocab,pad_idx
        self.copy_new += ['vocab', 'pad_idx']
        self.loss_func = CrossEntropyFlat()

    def get(self, i):
        ids = self.ids[i]
        fp = self.fps[i]
        return FpSmiles(ids, self.vocab.textify(ids, sep=''), fp)

    def reconstruct(self, ids:Tensor, fp:Tensor):
        return FpSmiles(ids, self.vocab.textify(ids), fp)

class DesmilesLoader():
    "Create a dataloader for desmiles."
    def __init__(self, dataset:FpSmilesList, bs:int=64, vocab=None, sampler=None, shuffle=False, drop_last=False):
        # shuffle an drop_last are required by fastai as arguments.  We don't use them.
        self.dataset,self.bs = dataset,bs
        self.first,self.i,self.iter = True,0,0
        self.nb = dataset.ids.shape[0] // bs
        self.batch_size=bs
        self.n = len(self.dataset)
        self.batch_first=False
        self.vocab=vocab
        self.sampler=sampler
        self.init_kwargs = dict(bs=bs, vocab=vocab, sampler=sampler)

    def __iter__(self):
        self.i,self.iter = 0,0
        inds = np.arange(len(self.dataset))
        if self.sampler is None:
            inds = np.random.permutation(inds)
            # Get index of largest idsuence
            largest_seq_ind = np.argmax(np.sum(self.dataset.ids > 0, axis=1))
            # Find where this index got randomly permuted to
            current_ind = np.where(inds == largest_seq_ind)[0][0]
            # Get the current first index
            zero_ind = inds[0]
            # Flip the first index with the largest sequence index
            inds[current_ind] = zero_ind
            inds[0] = largest_seq_ind
        else:
            seq_lengths = (self.dataset.ids > 0).sum(axis=1)
            inds = np.asarray([x for x in self.sampler(seq_lengths, key=lambda x: seq_lengths[x], bs=self.bs)])
        inds = inds[:self.bs*self.nb].reshape(self.nb, self.bs)
        while self.iter<len(self):
            batch_inds = inds[self.iter]
            sequences = self.dataset.ids[batch_inds]
            threes = (3*np.ones(sequences.shape[0], dtype=np.int32))[:,None]
            sequences = np.concatenate((threes, sequences), axis=1)
            lengths = (sequences > 0).sum(axis=1)
            perm_inds = np.argsort(-lengths)
            lengths = lengths[perm_inds]
            sequences = sequences[perm_inds,:lengths[0]]
            if self.batch_first:
                sequences = torch.tensor(sequences, dtype=torch.long)
            else:
                sequences = torch.tensor(sequences.T, dtype=torch.long)
            fingerprints = torch.tensor(np.asarray(self.dataset.fps[batch_inds].todense(), dtype=np.float32)[perm_inds])
            lengths = torch.tensor(lengths, dtype=torch.long)
            self.iter += 1
            ## CHANGE THE x to sequences[:-1] and y to sequences[1:]
            yield ((sequences[:-1, :], fingerprints, lengths-1), sequences[1:,:].contiguous().view(-1))

    def __len__(self): return self.nb
