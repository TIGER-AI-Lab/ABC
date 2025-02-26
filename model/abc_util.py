import torch
import torch.nn.functional as F
import torch.distributed as dist

def compute_gathered_loss(q_emb, c_emb, temperature, label_smoothing=0.0):
    """
    Compute the loss by gathering across GPUs.
    Make sure that the first [batch_size] entries in c_emb conrenspond to the 'correct' embeddings.
    """

    q_emb = q_emb.float()
    c_emb = c_emb.float()


    if dist.is_initialized():

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        c_global = [torch.zeros_like(c_emb) for _ in range(world_size)]
        dist.all_gather(c_global, c_emb)
        c_global[rank] = c_emb
        c_global[0], c_global[rank] = c_global[rank], c_global[0]
    
    else:
        c_global = [c_emb]

    c_global = torch.cat(c_global, dim=0)
    
    loss_global, acc_global = compute_contrastive_loss(q_emb, c_global, temperature=temperature, label_smoothing=label_smoothing)
    num_cand = torch.tensor([c_global.size(0)], device=loss_global.device)
    return loss_global, acc_global, num_cand

def get_mean_token_embed(input_ids, hidden_state, padding_token_id, instruction_mask=None):
     
    if instruction_mask is not None: # apply instruction mask
       hidden_state = instruction_mask.unsqueeze(-1)*hidden_state

    mask = (input_ids != padding_token_id).unsqueeze(-1)
    masked_states = mask*hidden_state
    mean_token_emb = torch.mean(masked_states,dim=1) # Average
    return mean_token_emb

def compute_contrastive_loss(q_embeds, p_embeds, temperature, label_smoothing=0.0):

    bs = q_embeds.size(0)

    score = temperature(torch.matmul(q_embeds, p_embeds.t()))
    sim_targets = torch.arange(bs).to(score.device)  # [bs]

    # compute loss
    loss = F.cross_entropy(score, sim_targets, label_smoothing=label_smoothing)
    _max_score, max_idxs = torch.max(score, 1)

    accuracy = (max_idxs == sim_targets).sum() / bs

    return loss, accuracy

def get_last_token_embed(input_ids, hidden_state, padding_token_id, instruction_mask=None):
    # Find the position of the last non-padding token for each sequence
    mask = input_ids != padding_token_id  # Create a mask where padding tokens are False
    last_token_pos = mask.sum(dim=1) - 1  # Get the index of the last non-padding token

    # Create a range tensor for batch indexing
    batch_size = input_ids.size(0)
    batch_range = torch.arange(batch_size, device=input_ids.device)

    # Extract the last token embedding for each sequence
    last_token_embeds = hidden_state[batch_range, last_token_pos]

    return last_token_embeds