"""
We also support batched embedding, where the use passes in lists of text and images, for effficent processing.
Note: all items in a batch must have the same combination of modalities i.e. a batch can be (text), (image) or (text, image).
Of course you can compare embeddings across modalities after embedding.
"""

import torch
base_model = "Qwen/Qwen2-VL-7B-Instruct"
pretrain_adapter = "TIGER-Lab/ABC-Qwen2VL-Pretrain"
instruction_adapter = "TIGER-Lab/ABC-Qwen2VL-Instruct"

from functional.embed_function import get_batched_embed_function
batched_embed_function = get_batched_embed_function(base_model, pretrain_adapter, instruction_adapter)

text_and_img_embed = batched_embed_function(image=["./examples/dog_and_cat.jpg"]*10, text=["what is the animal on the left?"]*10)
text_embed = batched_embed_function(text=["what is the animal on the left?"]*10)
image_embed = batched_embed_function(image=["./examples/dog_and_cat.jpg"]*10)

