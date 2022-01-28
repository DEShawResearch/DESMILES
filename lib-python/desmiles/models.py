from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models.awd_lstm import EmbeddingDropout, RNNDropout, WeightDropout, dropout_mask
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

__all__ = ['DesmilesCore', 'LinearDecoder', 'Desmiles', 'RecurrentDESMILES', 'FPEmbedder', 'EmbeddingToSMILES',
           'FingerprintEmbedderCore', 'get_desmiles_model', 'get_fp_to_embedding_model', 'get_embedded_fp_to_smiles_model']


class DesmilesCore(nn.Module):
    '''Core Piece of DESMILES
       Does everything except final softmax layer (LinearDecoder)
    '''

    initrange=0.1

    def __init__(self, vocab_sz:int, fp_emb_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int=0, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False, num_bits=4096):
        super().__init__()
        self.bs,self.qrnn,self.ndir = 1, qrnn,(2 if bidir else 1)
        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                             1, bidirectional=bidir, batch_first=False) for l in range(n_layers)]
        self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])
        self.linear_fp = [nn.Sequential(*bn_drop_lin(num_bits, fp_emb_sz, p=0.0, actn=torch.nn.Tanh())), nn.Sequential(*bn_drop_lin(fp_emb_sz, n_hid, p=0.1, actn=torch.nn.Tanh()))]
        self.linear_fp = torch.nn.ModuleList(self.linear_fp)
            

    def forward(self, input_seq, input_fp, lengths):

        sl,bs = input_seq.size()
        if bs!=self.bs:
            self.bs=bs
        self.reset()
        # Apply LinearBlocks on input_fp
        emb_fp = input_fp
        for linear_fp in self.linear_fp:
            emb_fp = linear_fp(emb_fp)
        raw_output = self.input_dp(self.encoder_dp(input_seq))
        raw_outputs,outputs = [],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output = pack_padded_sequence(raw_output, lengths, batch_first=False)
            if l == 0:
                raw_output, new_h = rnn(raw_output, (emb_fp.unsqueeze(0), emb_fp.unsqueeze(0)))
            else:
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            raw_output, lengths = pad_packed_sequence(raw_output, batch_first=False)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            if l == self.n_layers - 1: outputs.append(raw_output)
        #self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

class LinearDecoder(nn.Module):
    "To go on top of a DesmilesCore module and create a DESMILES Model."
    initrange=0.1
    def __init__(self, n_out:int, n_hid:int, output_p:float, tie_encoder:nn.Module=None, bias:bool=True):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output.contiguous().view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


class Desmiles(nn.Module):
    "Combines DesmilesCore with LinearDecoder for a full model"
    def __init__(self, desmiles_rnn_core, linear_decoder):
        super().__init__()
        self.desmiles_rnn_core = desmiles_rnn_core
        self.linear_decoder = linear_decoder

    def reset(self):
        for c in [self.desmiles_rnn_core, self.linear_decoder]:
            if hasattr(c, 'reset'):
                c.reset()

    def forward(self, input_seq, input_fp, lengths):
        output = self.desmiles_rnn_core(input_seq, input_fp, lengths.detach().cpu())
        output = self.linear_decoder(output)
        return output

    def __getitem__(self,key):
        if key == 0:
            return self.desmiles_rnn_core
        elif key == 1:
            return self.linear_decoder
        else:
            raise ValueError("Indexing only supports 0 or 1")


def get_desmiles_model(vocab_sz:int, fp_emb_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int=0, tie_weights:bool=True,
                       qrnn:bool=False, bias:bool=True, bidir:bool=False, output_p:float=0.4, hidden_p:float=0.2, input_p:float=0.6,
                       embed_p:float=0.1, weight_p:float=0.5, num_bits=4096)->nn.Module:
    "Create a full DESMILES model."
    rnn_enc = DesmilesCore(vocab_sz, fp_emb_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, qrnn=qrnn, bidir=bidir,
                              hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p, num_bits=4096)
    enc = rnn_enc.encoder if tie_weights else None
    model = Desmiles(rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias))
    model.reset()
    return model


class RecurrentDESMILES(nn.Module):
    '''
    RecurrentDESMILES is a reimplimentation of DESMILES model which provides separate functions
    for the fingerprint embedding and the decoding. This is used heavily in decoding methods 
    such as beam search and A*
    '''
    initrange=0.1
    def __init__(self, desmiles, fp_embedding_layers=(0, 1)):
        super().__init__()
        self.desmiles = desmiles
        self.fp_embedding_layers = nn.ModuleList([desmiles[0].linear_fp[layer] for layer in fp_embedding_layers])
        self.embedding = desmiles[0].encoder
        self.rnns = nn.ModuleList([rnn for rnn in desmiles[0].rnns])
        self.final_layer = desmiles[1].decoder
        self.emb_sz = desmiles[0].emb_sz
        self.nhid = desmiles[0].n_hid
        self.nlayers = len(self.rnns)

    def embed_fingerprints(self, fps):
        assert next(self.desmiles.parameters()).device == fps.device
        self.bs = fps.shape[0]
        self.reset() # reset the hidden state every time we get a new fingerprint
        with torch.no_grad():
            embedded = fps
            for layer in self.fp_embedding_layers:
                embedded = layer(embedded)
            self.hiddens[0] = (embedded.unsqueeze(0), embedded.unsqueeze(0))

    def forward(self, seq):
        sl,bs = seq.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        lengths = (seq > 0).sum(dim=0)
        with torch.no_grad():
            output = self.embedding(seq)
            output = pack_padded_sequence(output, lengths.detach().cpu())
            for i, (rnn, hidden) in enumerate(zip(self.rnns, self.hiddens)):                
                output, hidden = rnn(output, hidden)
                self.hiddens[i] = hidden
            output, lengths = pad_packed_sequence(output)
            return self.final_layer(output).transpose(0,1)


    def one_hidden(self, l):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nh = (self.nhid if l != self.nlayers - 1 else self.emb_sz)
        return torch.zeros(1, self.bs, nh).to(device) 

    def reset(self):
        self.hiddens = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.nlayers)]

    def set_hiddens(self, hiddens):
        assert len(hiddens) == len(self.hiddens)
        self.hiddens = hiddens

    def select_hidden(self, idxs):
        self.hiddens = [(h[0][:,idxs,:],h[1][:,idxs,:]) for h in self.hiddens]
        self.bs = len(idxs)

#################################################################################
########################### Code to Split Model #################################
#################################################################################


class FPEmbedder(nn.Module):
    """ Maps fingerprints to their pretrained embedding space
        For simplicity of loading weights from DESMILES, I create a copy of DESMILESRNNCore
        and just replace the forward() method
    """
    initrange=0.1
    def __init__(self, vocab_sz:int, fp_emb_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int=0, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False, num_bits=4096, first_layer=False):
        super().__init__()
        self.first_layer = first_layer
        self.bs,self.qrnn,self.ndir = 1, qrnn,(2 if bidir else 1)
        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                             1, bidirectional=bidir, batch_first=False) for l in range(n_layers)]
        self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])
        self.linear_fp = [nn.Sequential(*bn_drop_lin(num_bits, fp_emb_sz, p=0.0, actn=torch.nn.Tanh())), nn.Sequential(*bn_drop_lin(fp_emb_sz, n_hid, p=0.1, actn=torch.nn.Tanh()))]
        self.linear_fp = torch.nn.ModuleList(self.linear_fp)


    def forward(self, input_fp):
        if self.first_layer:
            return self.linear_fp[0](input_fp)

        for layer in self.linear_fp:
            input_fp = layer(input_fp)
        return input_fp

class EmbeddingToSMILES(nn.Module):
    """Maps embedding space of fingerprints to SMILES
       For simplicity of loading weights from DESMILES, I create a copy of DESMILSE and just replace
       the forward() method
    """
    def __init__(self, desmiles_rnn_core, linear_decoder):
        super().__init__()
        self.desmiles_rnn_core = desmiles_rnn_core
        self.linear_decoder = linear_decoder

    def reset(self):
        for c in [self.desmiles_rnn_core, self.linear_decoder]:
            if hasattr(c, 'reset'):
                c.reset()

    def forward(self, input_seq, input_fp, lengths):
        output = self.desmiles_rnn_core(input_seq, input_fp, lengths.detach().cpu())
        output = self.linear_decoder(output)
        return output

    def __getitem__(self,key):
        if key == 0:
            return self.desmiles_rnn_core
        elif key == 1:
            return self.linear_decoder
        else:
            raise ValueError("Indexing only supports 0 or 1")


class FingerprintEmbedderCore(nn.Module):
    """Maps embedding space of fingerprints to SMILES
       For simplicity of loading weights from DESMILES, I create a copy of DESMILSE and just replace
       the forward() method
    """
    
    initrange=0.1

    def __init__(self, vocab_sz:int, fp_emb_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int=0, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False, num_bits=4096, first_layer=True):
        super().__init__()
        self.first_layer = first_layer
        self.bs,self.qrnn,self.ndir = 1, qrnn,(2 if bidir else 1)
        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                             1, bidirectional=bidir, batch_first=False) for l in range(n_layers)]
        self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])
        self.linear_fp = [nn.Sequential(*bn_drop_lin(num_bits, fp_emb_sz, p=0.0, actn=torch.nn.Tanh())), nn.Sequential(*bn_drop_lin(fp_emb_sz, n_hid, p=0.1, actn=torch.nn.Tanh()))]
        self.linear_fp = torch.nn.ModuleList(self.linear_fp)
            

    def forward(self, input_seq, input_fp, lengths):

        sl,bs = input_seq.size()
        if bs!=self.bs:
            self.bs=bs
        self.reset()
        # Apply LinearBlocks on input_fp
        emb_fp = input_fp
        if self.first_layer:
            for linear_fp in self.linear_fp[1:]:
                emb_fp = linear_fp(emb_fp)
        raw_output = self.input_dp(self.encoder_dp(input_seq))
        raw_outputs,outputs = [],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output = pack_padded_sequence(raw_output, lengths, batch_first=False)
            if l == 0:
                raw_output, new_h = rnn(raw_output, (emb_fp.unsqueeze(0), emb_fp.unsqueeze(0)))
            else:
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            raw_output, lengths = pad_packed_sequence(raw_output, batch_first=False)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            if l == self.n_layers - 1: outputs.append(raw_output)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

def get_fp_to_embedding_model(desmiles, first_layer=True, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fp_emb_sz = desmiles[0].linear_fp[0][1].out_features
    emb_sz = desmiles[0].emb_sz
    num_bits = desmiles[0].linear_fp[0][1].in_features
    n_tok = desmiles[0].encoder.weight.size()[0]
    nhid = desmiles[0].rnns[0].module.hidden_size
    nlayers = len(desmiles[0].rnns)
    pad_token = desmiles[0].encoder.padding_idx
    fp_to_embedding = FPEmbedder(n_tok, fp_emb_sz, emb_sz, nhid, nlayers, pad_token=pad_token, first_layer=first_layer)
    fp_to_embedding_dict = fp_to_embedding.state_dict()
    pretrained_dict = desmiles[0].state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in fp_to_embedding_dict}
    assert len(pretrained_dict.keys()) == len(fp_to_embedding_dict.keys())
    fp_to_embedding_dict.update(pretrained_dict)
    fp_to_embedding.load_state_dict(fp_to_embedding_dict)
    fp_to_embedding.eval()
    return fp_to_embedding.to(device)

def get_embedded_fp_to_smiles_model(desmiles, first_layer=True, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fp_emb_sz = desmiles[0].linear_fp[0][1].out_features
    emb_sz = desmiles[0].emb_sz
    num_bits = desmiles[0].linear_fp[0][1].in_features
    n_tok = desmiles[0].encoder.weight.size()[0]
    nhid = desmiles[0].rnns[0].module.hidden_size
    nlayers = len(desmiles[0].rnns)
    pad_token = desmiles[0].encoder.padding_idx
    embedded_fp_to_encoded_core = FingerprintEmbedderCore(n_tok, fp_emb_sz, emb_sz, nhid, nlayers, pad_token=pad_token, first_layer=first_layer)
    trained_dict = desmiles[0].state_dict()
    embedded_fp_to_encoded_core_dict = embedded_fp_to_encoded_core.state_dict()
    pretrained_dict = desmiles[0].state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in embedded_fp_to_encoded_core_dict}
    assert len(pretrained_dict.keys()) == len(embedded_fp_to_encoded_core_dict.keys())
    embedded_fp_to_encoded_core_dict.update(pretrained_dict)
    embedded_fp_to_encoded_core.load_state_dict(embedded_fp_to_encoded_core_dict)
    linear_decoder = desmiles[1]
    embedded_fp_to_encoded = EmbeddingToSMILES(embedded_fp_to_encoded_core, linear_decoder)
    embedded_fp_to_encoded.eval()
    return embedded_fp_to_encoded.to(device)
