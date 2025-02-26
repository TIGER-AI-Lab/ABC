import torch
base_model = "Qwen/Qwen2-VL-7B-Instruct"
pretrain_adapter = "TIGER-Lab/ABC-Qwen2VL-Pretrain"
instuct_adapter = "TIGER-Lab/ABC-Qwen2VL-Instruct"


from functional.embed_function import get_embed_function
embed_function, model = get_embed_function(base_model, pretrain_adapter, instuct_adapter)
