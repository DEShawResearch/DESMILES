import torch
import torch.nn.functional as F
import queue
from itertools import count
import numpy as np
from ..models import RecurrentDESMILES

class AstarTree:
    def __init__(self, fp, desmiles, max_length=30, max_branches=5000):
        assert type(fp) is torch.Tensor
        assert type(desmiles) is RecurrentDESMILES
        assert next(desmiles.parameters()).device == fp.device
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        self.fp = fp
        self.desmiles = desmiles

        self.max_length = max_length
        self.max_branches = max_branches
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unique_identifier = count() # counter breaks ties between equivalent probabilities
        self.non_leaf_queue = queue.PriorityQueue()
        self.leaf_queue = queue.PriorityQueue()

        self.desmiles.embed_fingerprints(fp) # initialize rnn_desmiles by embedding the fingerprint
        root = self.initialize_root_node()

        self.non_leaf_queue.put((torch.tensor(0.0, device=self.device), next(self.unique_identifier), root))
        self.leaf_queue.put((torch.tensor(float("inf"), device=self.device), next(self.unique_identifier), root))
        self.num_expand = 1
        self.num_branches = 0


    @classmethod
    def create_from_astar_tree(cls, other):
        astar_tree = cls(other.fp, other.desmiles, max_length=other.max_length, max_branches=other.max_branches)
        astar_tree.non_leaf_queue = other.non_leaf_queue
        astar_tree.leaf_queue = other.leaf_queue
        return astar_tree

    def initialize_root_node(self):
        root = torch.zeros(self.max_length, dtype=torch.long, device=self.device)
        root[0] = 3
        return root

    def return_leaf_node(self):
        score, _, seq = self.leaf_queue.get()  # pop the top, decode and yield.
        self.last_leaf_node = seq
        return score, seq

    def __next__(self):
        # If no more room for expansion
        while True:
            if self.num_branches > self.max_branches:
                if self.leaf_queue.qsize() > 0:
                    return self.return_leaf_node()
                else:
                    return np.inf, self.last_leaf_node
            # Return a leaf node if it has the best score
            if self.leaf_queue.queue[0][0] < self.non_leaf_queue.queue[0][0]:
                return self.return_leaf_node()
            # Otherwise, purge queue and branch
            self.purge_queue() # purge non-leaf queue if necessary
            self.branch_and_bound()

    def branch_and_bound(self):
        # get nodes to expand
        seqs, scores = self.branch()
        #seqs, scores = self.post_branch_callback(seqs, scores)
        self.bound(seqs, scores)
        self.num_branches += 1


    def branch(self):
        # Create scores and sequences for the num_expand nodes we will expand
        # it doesn't actually do the branching.
        scores = []
        seqs = []
        num_nodes = self.non_leaf_queue.qsize()
        while(len(scores) < min(self.num_expand, num_nodes)):
            score, _, seq = self.non_leaf_queue.get()
            scores.append(score)
            seqs.append(seq)
        scores = torch.tensor(scores, device=self.device)
        seqs = torch.stack(seqs)
        return seqs, scores

    def bound(self, seqs, scores):
        # Perform the branch operation and then bound the results:
        with torch.no_grad():
            # Clone hidden states which are the embedded fingerprints only.
            # This only needs as argument the len(seqs) [or seq.shape[0]]
            hiddens = self.clone_hidden_states(seqs)
            # Get the probabilities for all the children
            # The call to get_log_probabilities will overwrite the hidden states based on the sequences.
            log_probs = self.get_log_probabilities(seqs)
            # reset the hidden states to those of the embedded fingerprints.
            self.desmiles.hiddens = hiddens
            seqs, scores = self.get_children(seqs, log_probs, scores)
            self.add_children_to_queue(seqs, scores)


    def add_children_to_queue(self, seqs, scores):
        # sort scores and grab the first max_branches to add to the two separate queues.
        sort_idx = self.sort_scores(scores, self.max_branches)
        scores = scores[sort_idx]
        seqs = seqs[sort_idx]  # this is a 2D tensor with dimensions: (8000 x num_expanded_children) x 30 (padded sequences)
        is_leaf_node = self.are_children_leaf_nodes(seqs)
        for i, (score, child) in enumerate(zip(scores[is_leaf_node].tolist(), seqs[is_leaf_node])):
            self.leaf_queue.put((score, next(self.unique_identifier), child))
        for i, (score, child) in enumerate(zip(scores[~is_leaf_node].tolist(), seqs[~is_leaf_node])):
            self.non_leaf_queue.put((score, next(self.unique_identifier), child))

    def are_children_leaf_nodes(self, children):
        # the -1 is a hack for using the index in the last_chars line.  Could revert back to actual legths.
        lengths = (children > 0).sum(dim=1) - 1
        last_chars = torch.tensor([child[length] for child, length in zip(children, lengths)], device=self.device)
        # last_chars == 0 means a pad character was chosen. for now I call this a leaf node so that it is not expanded further
        # The special characters are hard coded here:
        last_char_is_stop = (last_chars == 1) | (last_chars == 2) | (last_chars == 0)
        # this leaf node check only runs up to 30 elements, hardcoded (differs from earlier leaf_node check).
        is_leaf_node = (((lengths + 1) > 3) & last_char_is_stop) | ((lengths + 1) == 30)
        return is_leaf_node

    @staticmethod
    def sort_scores(scores, max_branches):
        return torch.sort(scores)[1][:max_branches]

    def get_children(self, seqs, log_probs, parent_nlps):
        lengths = (seqs > 0).sum(dim=1)
        children = torch.arange(log_probs.size(1), device=self.device, dtype=torch.long)[None,:].expand(seqs.shape[0], log_probs.size(1))
        new_seqs = []
        for seq, child, length in zip(seqs, children, lengths):
            seq = seq.expand(child.size(0), seq.size(0))
            seq=torch.cat((seq[:,:length], child[:,None], seq[:,length:-1]), dim=1)
            new_seqs.append(seq)
        new_seqs = torch.stack(new_seqs)
        new_seqs = new_seqs.reshape(-1, new_seqs.shape[-1])
        scores = (parent_nlps[:,None] - log_probs).reshape(-1)
        return new_seqs, scores
    
    def clone_hidden_states(self, seqs):
        num_sequences = seqs.shape[0]
        if num_sequences != self.desmiles.bs:
            self.desmiles.select_hidden(torch.zeros(num_sequences, dtype=torch.long))
            self.desmiles.bs = num_sequences
        hiddens = [(h[0].clone(),h[1].clone()) for h in self.desmiles.hiddens]
        return hiddens

    def get_log_probabilities(self, seqs):
        logits = self.desmiles(seqs.transpose(0,1))
        lengths = (seqs > 0).sum(dim=1) - 1
        # get the energy corresponding to the next token (specified by lengths)
        logits = torch.stack([logits[i,l] for i,l in zip(np.arange(lengths.shape[0]), lengths.tolist())])
        assert(logits.dim() == 2)
        return F.log_softmax(logits, dim=1)

    def purge_queue(self, downto=10000, maxsize=1000000):  ## will need to update to also keep mol_queue mols.
        if self.non_leaf_queue.qsize() > maxsize:
            print("PURGING")
            q2 = queue.PriorityQueue()
            ## Get top elements into new queue
            for i in range(downto):
                q2.put(self.non_leaf_queue.get())
            self.non_leaf_queue = q2


class AstarTreeParallel(AstarTree):
    def __init__(self, fp, desmiles, max_length=30, max_branches=5000, num_expand=1):
        super().__init__(fp, desmiles, max_length=30, max_branches=5000)
        self.num_expand = num_expand

    def branch(self):
        seqs, scores = super().branch()
        return self.sort_by_length(seqs, scores)

    @staticmethod
    def sort_by_length(seqs, scores):
        lengths = (seqs > 0).sum(dim=1)
        length_idx = AstarTreeParallel.sort_lengths(lengths)
        seqs = seqs[length_idx]
        scores = scores[length_idx]
        return seqs, scores

    @staticmethod
    def sort_lengths(lengths):
        return torch.sort(-lengths)[1]



class AstarTreeParallelNotSafe:
    def __init__(self, fp, desmiles, max_length=30, max_branches=5000, num_expand=1):
        assert type(fp) is torch.Tensor
        assert type(desmiles) is RecurrentDESMILES
        assert next(desmiles.parameters()).device == fp.device
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        self.fp = fp
        self.desmiles = desmiles

        self.max_length = max_length
        self.max_branches = max_branches
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unique_identifier = count() # counter breaks ties between equivalent probabilities
        self.non_leaf_queue = queue.PriorityQueue()
        self.leaf_queue = queue.PriorityQueue()

        self.desmiles.embed_fingerprints(fp) # initialize rnn_desmiles by embedding the fingerprint
        root = self.initialize_root_node()

        self.non_leaf_queue.put((torch.tensor(0.0, device=self.device), next(self.unique_identifier), root))
        self.leaf_queue.put((torch.tensor(float("inf"), device=self.device), next(self.unique_identifier), root))
        self.num_expand = 1

        self.num_expand = num_expand
        self.node_to_hiddens = {}
        score, ident, root = self.non_leaf_queue.get()
        self.node_to_hiddens[root] = (self.clone_hiddens(), 0)
        self.non_leaf_queue.put((score, ident, root))
        self.num_expand = num_expand
        self.num_branches = 0

    def initialize_root_node(self):
        # The sequences are kept in reverse order from right to left starting from the last element;
        # This is only happening for the AstarTreeParallelNotSafe version.
        # In this way we don't need to look at the length vectors since some sequences will have differing lengths.
        # In the memory safe way, we can keep the sequences right padded.  
        root = torch.zeros(self.max_length, dtype=torch.long, device=self.device)
        root[-1] = 3
        return root

    def clone_hiddens(self):
        return [(h[0].clone(),h[1].clone()) for h in self.desmiles.hiddens]

    def __next__(self):
        # If no more room for expansion
        while True:
            # In this version of Astar we don't keep expanding if we've reached the max branches.
            # This differs from the logic of the early astar algorithm used in the paper.
            if self.num_branches > self.max_branches:
                if self.leaf_queue.qsize() > 0:
                    return self.return_leaf_node()
                else:
                    return np.inf, self.last_leaf_node
            # Return a leaf node if it has the best score
            if self.leaf_queue.queue[0][0] < self.non_leaf_queue.queue[0][0]:
                return self.return_leaf_node()
            # Otherwise, purge queue and branch
            self.purge_queue() # purge non-leaf queue if necessary
            self.branch_and_bound()

    def branch_and_bound(self):
        # get nodes to expand
        seqs, scores = self.branch()
        #seqs, scores = self.post_branch_callback(seqs, scores)
        self.bound(seqs, scores)
        self.num_branches += 1


    def branch(self):
        # Prepare for the branching operation
        # This branch is more complex than the plain Astar because it handles hidden states.
        # Collect the scores, seqs, hiddens, etc for up to num_expand nodes
        scores = []
        seqs = []
        hiddens = []
        # When you branch a node every child has its own hidden state branching from the same parent hidden state
        # this index maps back to the hidden state of the parent.
        hidden_idxs = []
        num_nodes = self.non_leaf_queue.qsize()
        # This is a dictionary from the node tensor to a hidden state and a hidden_idx.
        node_to_hiddens = self.node_to_hiddens
        while(len(scores) < min(self.num_expand, num_nodes)):
            score, _, seq = self.non_leaf_queue.get()
            scores.append(score)
            seqs.append(seq)
            hidden, idx = node_to_hiddens[seq]
            hiddens.append(hidden)
            hidden_idxs.append(idx)
        scores = torch.tensor(scores, device=self.device)
        seqs = torch.stack(seqs)
        # select all the parent hidden states
        hiddens = AstarTreeParallelNotSafe.select_all_hiddens(hiddens, hidden_idxs)
        # concatenate all the states so we can batch evaluate them.
        hiddens = AstarTreeParallelNotSafe.concat_hiddens(hiddens)
        # set the hidden states
        self.desmiles.hiddens = hiddens
        # set the batch size (seqs legnth is same as length of hiddens[0][0]; first layer; cell state of hidden)
        self.desmiles.bs = seqs.shape[0]
        return seqs, scores

    def bound(self, seqs, scores):
        with torch.no_grad():
            # hidden states are ready, so this only passes the last elements to the desmiles model
            log_probs = self.get_log_probabilities(seqs[:,-1].unsqueeze(0))
            # Get an index for what parent that child was from.
            # Important for getting the hidden states back.
            seq_idx = self.get_seq_idx(log_probs)
            # Make the sequences and the scores of the children
            seqs, scores = self.get_children(seqs, log_probs, scores)
            self.add_children_to_queue(seqs, scores, seq_idx)

    def get_log_probabilities(self, seqs):
        logits = self.desmiles(seqs)
        return F.log_softmax(logits[:,-1], dim=-1)


    def add_children_to_queue(self, seqs, scores, seq_idxs):
        node_to_hiddens = self.node_to_hiddens
        hiddens = self.clone_hiddens()
        sort_idx = AstarTreeParallelNotSafe.sort_scores(scores, self.max_branches)
        scores = scores[sort_idx]
        seqs = seqs[sort_idx]
        seq_idxs = seq_idxs[sort_idx]
        is_leaf_node = self.are_children_leaf_nodes(seqs)
        for i, (score, child, seq_idx) in enumerate(zip(scores[is_leaf_node].tolist(), seqs[is_leaf_node], seq_idxs[is_leaf_node].tolist())):
            node_to_hiddens[child] = (hiddens, seq_idx)
            self.leaf_queue.put((score, next(self.unique_identifier), child))
        for i, (score, child, seq_idx) in enumerate(zip(scores[~is_leaf_node].tolist(), seqs[~is_leaf_node], seq_idxs[~is_leaf_node].tolist())):
            node_to_hiddens[child] = (hiddens, seq_idx)
            self.non_leaf_queue.put((score, next(self.unique_identifier), child))

    @staticmethod
    def sort_scores(scores, max_branches):
        return torch.sort(scores)[1][:max_branches]


    def get_children(self, seqs, log_probs, parent_nlps):
        children = torch.arange(log_probs.size(1), device=self.device, dtype=torch.long)[None,:].expand(seqs.shape[0], log_probs.size(1))[:,:,None]
        children = torch.cat([seqs[:,None,:].expand(seqs.size(0), log_probs.size(1), seqs.size(1)), children],dim=2)
        children = children.reshape(-1, children.shape[-1])[:,1:]
        scores = (parent_nlps[:,None] - log_probs).reshape(-1)
        return children, scores

    def are_children_leaf_nodes(self, children):
        lengths = (children > 0).sum(dim=1) - 1
        last_chars = children[:,-1]
        last_char_is_stop = (last_chars == 1) | (last_chars == 2)
        is_leaf_node = (((lengths + 1) > 3) & last_char_is_stop) | ((lengths + 1) == 30) | (last_chars == 0) 
        return is_leaf_node

    @staticmethod
    def select_all_hiddens(hiddens, idxs):
        return [AstarTreeParallelNotSafe.select_hiddens(hidden, idx) for hidden, idx in zip(hiddens, idxs)]

    @staticmethod
    def select_hiddens(hiddens, idx, move_to_device=False):
        return [(h[0][:,idx:idx+1,:],h[1][:,idx:idx+1,:]) for h in hiddens]

    def get_seq_idx(self, log_probs):
        return torch.arange(log_probs.size(0), dtype=torch.long, device=self.device).unsqueeze(1).expand(log_probs.shape[0], log_probs.shape[1]).reshape(-1)


    @staticmethod
    def concat_hiddens(hiddens):
        layers = []
        for layer in range(len(hiddens[0])):
            hidden_states = torch.cat([h[layer][0] for h in hiddens], dim=1)
            cell_states = torch.cat([h[layer][1] for h in hiddens], dim=1)
            layers.append((hidden_states, cell_states))
        return layers

    def purge_queue(self, downto=10000, maxsize=1000000): 
        if self.non_leaf_queue.qsize() > maxsize:
            q2 = queue.PriorityQueue()
            ## Get top elements into new queue
            for i in range(downto):
                q2.put(self.non_leaf_queue.get())
            self.non_leaf_queue = q2

    def return_leaf_node(self):
        score, _, seq = self.leaf_queue.get()  # pop the top, decode and yield.
        self.last_leaf_node = seq
        return score, seq

def left_pad_to_right_pad(seqs):
    lengths = (seqs > 0).sum(dim=1)
    ns, sl = seqs.shape
    new_seqs = torch.zeros_like(seqs)
    for i,(s,l) in enumerate(zip(seqs, lengths)):
        new_seqs[i,:l] = s[s>0]
    return new_seqs

class AstarTreeParallelHybrid(AstarTreeParallelNotSafe):
    @staticmethod
    def make_queue_with_right_padding(queue):
        scores = []
        ids = []
        tensors = []
        for score, i, t in queue.queue:
            scores.append(score)
            ids.append(i)
            tensors.append(t)
        tensors = left_pad_to_right_pad(torch.stack(tensors))
        new_queue = [(s,i,t) for s,i,t in zip(scores, ids, tensors)]
        queue.queue = new_queue
        return queue

    def __next__(self):
        safe_branch_thresh = 500
        # If no more room for expansion
        while True:
            if self.num_branches == safe_branch_thresh:
                print("Switching to memory safe queue!")
                self.purge_queue(maxsize=10000)
                self.non_leaf_queue = AstarTreeParallelHybrid.make_queue_with_right_padding(self.non_leaf_queue)
                self.leaf_queue = AstarTreeParallelHybrid.make_queue_with_right_padding(self.leaf_queue)

                self.safe_astar = AstarTreeParallel.create_from_astar_tree(self)
                self.safe_astar.num_expand = self.num_expand
                self.safe_astar.desmiles.embed_fingerprints(self.fp) # initialize rnn_desmiles by embedding the fingerprint
                del self.node_to_hiddens
                torch.cuda.empty_cache()
            if self.num_branches < safe_branch_thresh:
                if self.num_branches > self.max_branches:
                    if self.leaf_queue.qsize() > 0:
                        return self.return_leaf_node()
                    else:
                        return np.inf, self.last_leaf_node
                # Return a leaf node if it has the best score
                if self.leaf_queue.queue[0][0] < self.non_leaf_queue.queue[0][0]:
                    return self.return_leaf_node()
                # Otherwise, purge queue and branch
                self.purge_queue() # purge non-leaf queue if necessary
                self.branch_and_bound()
            else:
                self.num_branches += 1
                return next(self.safe_astar)

########################
### OLD ASTAR CODE
########################

def get_astar_tree(fp, model, decoder, isLeafNode, max_length=30, max_branches=5000):
    assert type(fp) is torch.Tensor
    #import pdb; pdb.set_trace()
    partial_mol_queue = queue.PriorityQueue()
    # Initialize string with (3)
    current_string = [3]
    partial_mol_queue.put((0, tuple(current_string)))
    # Add and initialize a Complete molecule queue
    mol_queue = queue.PriorityQueue()
    mol_queue.put((np.inf, tuple(current_string)))
    num_branches = 1
    last_smiles_string = ''
    last_smiles_vector = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    while True:
        if num_branches > max_branches:  # no more room for expansion
            if mol_queue.qsize() > 4:  # if we have enough addional molecules (which we should have)
                score, smiles_tup = mol_queue.get()  # pop the top, decode and yield.
                smiles_string = decoder(list(smiles_tup))
                last_smiles_string = smiles_string
                last_smiles_vector = np.asarray(smiles_tup)
                yield score, smiles_string, last_smiles_vector, num_branches
            else:
                yield np.inf, last_smiles_string, last_smiles_vector, num_branches
        partial_mol_queue = purgedQueue(partial_mol_queue)
        # grab the queue with the higher priority
        que_to_pop = partial_mol_queue if partial_mol_queue.queue[0][0] < mol_queue.queue[0][0] else mol_queue
        score, smiles_tup = que_to_pop.get()
        if isLeafNode(smiles_tup):
            smiles_string = decoder(list(smiles_tup))
            last_smiles_string = smiles_string
            last_smiles_vector = np.asarray(smiles_tup)
            yield score, smiles_string, last_smiles_vector, num_branches
        else:
            num_branches += 1
            children = getChildren(fp, list(smiles_tup), model, score, device=device)
            for i, (score, child) in enumerate(children):
                if (i==1) or (i==2):
                    # change if (len(child) < 3) or (child[0] == child[-1]):
                    if (len(child) < 4) or (child[1] == child[-1]):
                        partial_mol_queue.put((score, tuple(child)))
                        continue
                    mol_queue.put((score, tuple(child)))  # the begin and end subSMILES
                else:
                    partial_mol_queue.put((score, tuple(child)))


def getChildren(fp, smiles_list, model, parent_score, max_length=30, device="cpu"):
    import torch.nn.functional as F
    lengths = [len(smiles_list)]
    inp_smiles = np.zeros((max_length, 1), dtype=np.int32)
    inp_smiles[:len(smiles_list), 0] = np.asarray(smiles_list)
    fp = fp[None]
    with torch.no_grad():
        energies, _, _ = model(torch.tensor(inp_smiles, dtype=torch.long, device=device), fp, torch.tensor(lengths, dtype=torch.long, device=device))
        log_probs = F.log_softmax(energies[-1],dim=-1).data
    children = []
    for i, log_prob in enumerate(log_probs):
        child = smiles_list.copy()
        child.append(i)
        children.append((parent_score - log_prob, child))
    return children


def purgedQueue(q, downto=10000, maxsize=1000000):  ## will need to update to also keep mol_queue mols.
    if q.qsize() > maxsize:
        q2 = queue.PriorityQueue()
        ## Get top elements into new queue
        for i in range(downto):
            q2.put(q.get())
        return q2
    return q
