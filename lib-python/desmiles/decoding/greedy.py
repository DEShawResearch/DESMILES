import torch
import numpy as np
import torch.nn.functional as F

def beam_search(rnn_desmiles, fp, beam_sz=100, max_tokens=30):
    assert beam_sz <= rnn_desmiles.embedding.weight.shape[0], "Beam Size must be smaller than number of tokens"
    assert fp.shape[0] == 1, "Currently must be tensor of size (1, fp_size)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    leaf_nodes = []
    with torch.no_grad():
        rnn_desmiles.embed_fingerprints(fp) # initialize rnn_desmiles by embedding the fingerprint
        nodes = torch.tensor([[3]]).to(device) # root node [3]
        xb = nodes.clone() # the last token of each node. this is what actually runs through the model
        scores = torch.tensor([0.0]).to(device) # scores for the nodes (negative log probabilities)
        for t in range(max_tokens):
            out = F.log_softmax(rnn_desmiles(xb.transpose(0,1))[:,-1], dim=-1) # run the last token of each node through the model (maybe .contiguous())
            values, indices = out.topk(beam_sz, dim=-1) # get the top beam_sz scores for each node. this is a speed optimization since we will only keep beam_sz nodes in total
            scores = (-values + scores[:,None]).view(-1) # update scores (negative log probabilities) based on most recent conditional probability
            indices_idx = torch.arange(0,nodes.size(0))[:,None].expand(nodes.size(0), beam_sz).contiguous().view(-1).long()
            sort_idx = torch.sort(scores)[1][:beam_sz] # grab the indices of the top beam_sz scores, in pytorch >= 1.0 use scores.argsort() instead
            scores = scores[sort_idx] # sort the scores

            nodes = torch.cat([nodes[:,None].expand(nodes.size(0),beam_sz,nodes.size(1)),
                               indices[:,:,None].expand(nodes.size(0),beam_sz,1),], dim=2) # get the set of beam_sz * beam_sz nodes searched
            nodes = nodes.view(-1, nodes.size(2))[sort_idx]  # flatten them out and grab the top beam_sz 
            rnn_desmiles.select_hidden(indices_idx[sort_idx]) # update the hidden states of DESMILES based on which nodes we are keeping
            xb = nodes[:,-1][:,None] # update xb to be the last token of the nodes
            if nodes.shape[1] > 2:
                are_leaf_nodes = (nodes[:,-1] == 2) | (nodes[:,-1] == 1)
                for n,s in zip(nodes[are_leaf_nodes], scores[are_leaf_nodes]): # grab the leaf nodes and scores
                    leaf_nodes.append((n,s))
    leaf_nodes = [(n.cpu().numpy().tolist(),s) for (n,s) in leaf_nodes]
    leaf_nodes, scores = zip(*leaf_nodes)
    leaf_nodes = np.asarray(leaf_nodes)
    scores = np.asarray([s.item() for s in scores])
    sort_idx = np.argsort(scores)[:beam_sz]
    scores = scores[sort_idx]
    leaf_nodes = leaf_nodes[sort_idx]
    return leaf_nodes, scores
