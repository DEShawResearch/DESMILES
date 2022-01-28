'Model training for NLP'
from fastai.core import *
from fastai.torch_core import *
from fastai.basic_data import DataBunch
from fastai.basic_train import LearnerCallback, Learner
from fastai.text.learner import RNNLearner
from fastai.callbacks import annealing_linear, Scheduler
from .models import get_desmiles_model
from functools import partial
import torch.nn.functional as F

__all__ = ['desmiles_split', 'desmiles_model_learner', 'DesmilesLearner', 'OriginalFastaiOneCycleScheduler']


def desmiles_split(model:nn.Module) -> List[nn.Module]:
    '''
    Split a DESMILES `model` in groups for differential learning rates.
    This is currently never used but could be used in the future
    '''
    groups = [[rnn, dp] for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    groups.append([feedforward for feedforward in  model[0].linear_fp])
    groups.append([model[0].encoder, model[0].encoder_dp, model[1]])
    return groups


def desmiles_model_learner(data:DataBunch, fp_emb_sz:int=2000, emb_sz:int=400, nh:int=2000, nl:int=5, pad_token:int=0,
                           drop_mult:float=1., tie_weights:bool=True, bias:bool=True, qrnn:bool=False,
                           num_bits=4096, dropouts=(0.25, 0.1, 0.2, 0.02, 0.15), **kwargs) -> 'LanguageLearner':
    "Create a `DesmilesLearner` with a DESMILES model from `data`."
    assert(len(dropouts) == 5)
    dps = np.asarray(dropouts) * drop_mult
    vocab_size = len(data.vocab.itos)
    model = get_desmiles_model(vocab_size, fp_emb_sz, emb_sz, nh, nl, pad_token=pad_token, input_p=dps[0], output_p=dps[1],
                               weight_p=dps[2], embed_p=dps[3], hidden_p=dps[4], tie_weights=tie_weights, bias=bias, qrnn=qrnn, num_bits=num_bits)
    loss_func=partial(F.cross_entropy, ignore_index=0)
    learn = DesmilesLearner(data, model, split_func=desmiles_split, loss_func=loss_func, **kwargs)
    return learn


class DesmilesLearner(RNNLearner):

    def load_old(self, name:PathOrStr, device:torch.device=None, strict:bool=True, with_opt:bool=None, verbose:bool=False):
        '''
        Load the weights from an old (i.e. python3.6 and pytorch 0.4 DESMILES model)
        '''
        if device is None: device = self.data.device
        state = torch.load(self.path/self.model_dir/f'{name}.h5', map_location=device)
        if set(state.keys()) == {'model', 'opt'}:
            get_model(self.model).load_state_dict(state['model'], strict=strict)

            if ifnone(with_opt,True):
                if not hasattr(self, 'opt'): opt = self.create_opt(defaults.lr, self.wd)
                try:    self.opt.load_state_dict(state['opt'])
                except: pass
        else:
            if with_opt: warn("Saved filed doesn't contain an optimizer state.")
            sd = OrderedDict({self.replace_name(k):v for k,v in state.items()})
            if verbose:
                for k in self.model.state_dict().keys():
                    if k not in sd:
                        print(k, "not found")
            get_model(self.model).load_state_dict(sd, strict=strict)
        return self

    def replace_name(self, name):
        import re
        name = self.replace_sequential_name(name)
        layer_names = list(self.model.state_dict().keys())
        if "encoder_with_dropout.embed" in name:
            return name.replace("encoder_with_dropout.embed", "encoder_dp.emb")
        if ".module.weight_hh_l0_raw"  in name:
            return name.replace(".module","")
        if "linear_fp" in name and "lin.weight" in name:
            layer_number = int(re.search("linear_fp_(\d)", name).groups()[0])
            if name.replace(f"linear_fp_{layer_number}.lin.", f"linear_fp.{layer_number - 1}.1.") in layer_names:
                return name.replace(f"linear_fp_{layer_number}.lin.", f"linear_fp.{layer_number - 1}.1.")
            else:
                return name.replace(f"linear_fp_{layer_number}.lin.", f"linear_fp.{layer_number - 1}.2.")
        if "linear_fp" in name and "lin.bias" in name:
            layer_number = int(re.search("linear_fp_(\d)", name).groups()[0]) 
            if name.replace(f"linear_fp_{layer_number}.lin.", f"linear_fp.{layer_number - 1}.1.") in layer_names:
                return name.replace(f"linear_fp_{layer_number}.lin.", f"linear_fp.{layer_number - 1}.1.")
            else:
                return name.replace(f"linear_fp_{layer_number}.lin.", f"linear_fp.{layer_number - 1}.2.")
        if "linear_fp" in name and "bn.weight" in name:
            layer_number = int(re.search("linear_fp_(\d)", name).groups()[0])
            return f"desmiles_rnn_core.linear_fp.{layer_number - 1}.0.weight"
        if "linear_fp" in name and "bn.bias" in name:
            layer_number = int(re.search("linear_fp_(\d)", name).groups()[0])
            return f"desmiles_rnn_core.linear_fp.{layer_number - 1}.0.bias"
        if "linear_fp" in name and "bn.running_mean" in name:
            layer_number = int(re.search("linear_fp_(\d)", name).groups()[0])
            return f"desmiles_rnn_core.linear_fp.{layer_number - 1}.0.running_mean"
        if "linear_fp" in name and "bn.running_var" in name:
            layer_number = int(re.search("linear_fp_(\d)", name).groups()[0])
            return f"desmiles_rnn_core.linear_fp.{layer_number - 1}.0.running_var"
        return name


    @staticmethod
    def replace_sequential_name(name):
        if "0.encoder" in name:
            return name.replace("0.encoder", "desmiles_rnn_core.encoder")
        elif "0.rnns" in name:
            return name.replace("0.rnns", "desmiles_rnn_core.rnns")
        elif "0.linear" in name:
            return name.replace("0.linear", "desmiles_rnn_core.linear")
        elif "1.decoder" in name:
            return name.replace("1.decoder", "linear_decoder.decoder")
        else:
            raise ValueError("Could not find replacement for name")

class OriginalFastaiOneCycleScheduler(LearnerCallback):
    "Scheduler that mimics the original Fastai one-cycle learner"
    def __init__(self, learn:Learner, lr_max:float, moms:Floats=(0.8, 0.6), div_factor:float=10., frac_inc:float=0.5, frac_dec=0.49, tot_epochs:int=None, start_epoch:int=None):
        super().__init__(learn)
        self.lr_max,self.div_factor,self.frac_inc,self.frac_dec = lr_max,div_factor,frac_inc,frac_dec
        self.moms=tuple(listify(moms,2))
        if is_listy(self.lr_max): self.lr_max = np.array(self.lr_max)
        self.start_epoch, self.tot_epochs = start_epoch, tot_epochs

    def steps(self, *steps_cfg:StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [Scheduler(step, n_iter, func=func)
                for (step,(n_iter,func)) in zip(steps_cfg, self.phases)]

    def on_train_begin(self, n_epochs:int, epoch:int, **kwargs:Any)->None:
        "Initialize our optimization params based on our annealing schedule."
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.tot_epochs = ifnone(self.tot_epochs, n_epochs)
        n = len(self.learn.data.train_dl) * self.tot_epochs
        a1 = int(n * self.frac_inc)
        a2 = int(n * self.frac_dec)
        a3 = n - (a2 + a1)
        self.phases = ((a1, annealing_linear), (a2, annealing_linear), (a3, annealing_linear))
        low_lr = self.lr_max/self.div_factor
        self.lr_scheds = self.steps((low_lr, self.lr_max), (self.lr_max, low_lr), (low_lr, low_lr/(self.div_factor**2)))
        self.mom_scheds = self.steps(self.moms, (self.moms[1], self.moms[0]), (self.moms[0], self.moms[0]))
        self.opt = self.learn.opt
        self.opt.lr,self.opt.mom = self.lr_scheds[0].start,self.mom_scheds[0].start
        self.idx_s = 0
        return res
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take one step forward on the annealing schedule for the optim params."
        if train:
            if self.idx_s >= len(self.lr_scheds): return {'stop_training': True, 'stop_epoch': True}
            self.opt.lr = self.lr_scheds[self.idx_s].step()
            self.opt.mom = self.mom_scheds[self.idx_s].step()
            # when the current schedule is complete we move onto the next
            # schedule. (in 1-cycle there are two schedules)
            if self.lr_scheds[self.idx_s].is_done:
                self.idx_s += 1

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Tell Learner to stop if the cycle is finished."
        if epoch > self.tot_epochs: return {'stop_training': True}
