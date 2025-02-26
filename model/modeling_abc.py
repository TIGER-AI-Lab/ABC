from transformers import Qwen2VLForConditionalGeneration, LlavaNextForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from .abc_util import *
from torch import nn
from torch.nn import init

# Use custom iniatialization for linear layers to prevent very large values for bias.
class Linear(nn.Linear):

    def reset_parameters(self) -> None:
        init.eye_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

class MLP(nn.Module):

    def __init__(self, embed_size: int, hidden_size: int = None):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size if hidden_size is not None else embed_size

        # Initialize in high precision to prevent xavier init from underflowing
        self.linear_layer1 = Linear(self.embed_size, self.hidden_size, dtype=torch.float32)
        self.linear_layer2 = Linear(self.hidden_size, self.embed_size, dtype=torch.float32)
        self.act = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.linear_layer1(x)
        y = self.act(y)
        y = self.linear_layer2(y)
        out = x+y
        return out

class Temperature(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(0.07, requires_grad=True, dtype=torch.float32))        
    
    def forward(self, x):
        return x/(self.temp.float())

def get_temperature_for_logging(model):
    if isinstance(model.temperature, Temperature):
        return model.temperature.temp.data.clone().detach()
    else:
        return model.temperature.modules_to_save.default.temp.data.clone().detach()

class ABCqwen2VL(Qwen2VLForConditionalGeneration):
    
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    and MLP projection layer.
    """
    
    supports_gradient_checkpointing = True
    attn_mask = "bidirectional"
    instruction_mode = False

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mlp_head = MLP(config.hidden_size, hidden_size=4096)
        self.temperature = Temperature()
        
    def forward(self, inputs, return_outputs=False, return_prediction=False):

        query = inputs["query"]
        candidate = inputs["pos_cand"]
        instruction_mask = query.pop("instruction_mask", None)
        query.pop("labels", False)
        candidate.pop("labels", False)

        query_outputs : CausalLMOutputWithPast = super().forward(**query, output_hidden_states=True)
        
        # Use the base model to embed candidates in instruction mode
        if self.instruction_mode:
            with torch.no_grad(), self.get_peft_wrapper().disable_adapter():
                candidate_outputs : CausalLMOutputWithPast = super().forward(**candidate, output_hidden_states=True)    
        else:
            candidate_outputs : CausalLMOutputWithPast = super().forward(**candidate, output_hidden_states=True)
        
        # ensure logits computation was skipped for memory / speed
        # (or other LLM in used)
        assert(query_outputs.logits is None)
        assert(candidate_outputs.logits is None)

        q_eos_token_emb = get_mean_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0, instruction_mask=instruction_mask)
        c_eos_token_emb= get_mean_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)
        
        q_emb = self.mlp_head(q_eos_token_emb).float()
        c_emb = self.mlp_head(c_eos_token_emb).float()
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb = F.normalize(c_emb, dim=-1)
        
        # eval batch_size is num_gpus * eval_per_gpu
        loss, acc, num_cand = compute_gathered_loss(q_emb, c_emb, temperature=self.temperature, label_smoothing=0.1)

        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = get_temperature_for_logging(self)
            outputs["num_cand"] = num_cand

        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if (return_outputs or return_prediction) else loss
    
    # For unsupervised embedding using a pretrained model.
    def embed(self, inputs, instruction_mask = None):
        with torch.no_grad():
            hiddens : CausalLMOutputWithPast = super().forward(**inputs, output_hidden_states=True)
            averaged_hiddens = get_mean_token_embed(inputs["input_ids"], hiddens.hidden_states[-1], 0, instruction_mask=instruction_mask)
            proj = self.mlp_head(averaged_hiddens).float()
            return F.normalize(proj, dim=-1).detach()
        
    def inst_embed(self, inputs, is_cand):
        with torch.no_grad():
            if is_cand:
                with self.get_peft_wrapper().disable_adapter():
                    hiddens : CausalLMOutputWithPast = super().forward(**inputs, output_hidden_states=True)
            else:
                hiddens : CausalLMOutputWithPast = super().forward(**inputs, output_hidden_states=True)
                
            averaged_hiddens = get_mean_token_embed(inputs["input_ids"], hiddens.hidden_states[-1], 0, instruction_mask=None)
            proj = self.mlp_head(averaged_hiddens).float()
            return F.normalize(proj, dim=-1).detach()

MODEL_ARCHITECTURE = {
    "ABCQWEN": ABCqwen2VL,
 }