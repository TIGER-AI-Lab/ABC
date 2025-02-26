import torch
base_model = "Qwen/Qwen2-VL-7B-Instruct"
pretrain_adapter = "TIGER-Lab/ABC-Qwen2VL-Pretrain"
instruction_adapter = "TIGER-Lab/ABC-Qwen2VL-Instruct"

from functional.embed_function import get_embed_function
embed_function = get_embed_function(base_model, pretrain_adapter, instruction_adapter)

what_is_left_animal = embed_function(image="./examples/dog_and_cat.jpg", text="what is the animal on the left?")
what_is_right_animal = embed_function(image="./examples/dog_and_cat.jpg", text="what is the animal on the right?")

dog = embed_function(text="a dog")
cat = embed_function(text="a cat")

is_dog = torch.matmul(what_is_left_animal,dog.t())
is_cat = torch.matmul(what_is_left_animal,cat.t())

if is_dog > is_cat:
    print("the animal on the left is a dog")
else:
    print("the animal on the left is a cat")

is_dog = torch.matmul(what_is_right_animal,dog.t())
is_cat = torch.matmul(what_is_right_animal,cat.t())

if is_dog > is_cat:
    print("the animal on the right is a dog")
else:
    print("the animal on the right is a cat")
