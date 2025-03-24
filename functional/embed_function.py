import torch

# --  Efficency and Attn patches for tranfomers lib  ----------------------
from monkey_patch.qwen_attn_patch import monkey_patch_transformers_lib, unmask_attn_monkey_patch
monkey_patch_transformers_lib()
unmask_attn_monkey_patch()
# -------------------------------------------------------------------------

from qwen.vision_process import process_vision_info
import functools
from peft import PeftModel
from collections.abc import Mapping
from .input_templating import *

# Courtesy of the goods folks @ HF
def _prepare_input(data):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        cuda_tensor = data.cuda()
        return cuda_tensor
    return data

def get_model(base_model, pretrain_adapter, instruction_adapter):
        from model.modeling_abc import ABCqwen2VL
        base_model = ABCqwen2VL.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
        
        base_model.instruction_mode = True

        # Load and merge pretrain adapter
        pretrained_model = PeftModel.from_pretrained(base_model, pretrain_adapter)
        pretrained_model = pretrained_model.merge_and_unload()

        # Load instruction model
        model = PeftModel.from_pretrained(pretrained_model, instruction_adapter, adapter_name="instruct")
        
        # The forward method needs to be able to toggle LoRA
        setattr(model.get_base_model(), "get_peft_wrapper", lambda: model)
        model.to(torch.bfloat16).cuda()
        return model

def get_embed_function(base_model, pretrain_adapter, instruction_adapter):
        min_pixels = 256*28*28
        max_pixels = 1024*28*28
        from transformers import AutoProcessor
        
        model = get_model(base_model, pretrain_adapter, instruction_adapter)

        processor = AutoProcessor.from_pretrained(base_model,
                                                padding_side="right",
                                                use_fast=False,
                                                max_pixels=max_pixels,
                                                min_pixels=min_pixels)
        
    
        def embed(model, processor, image = None, text = None):

            conversation = None
            use_adapter = False

            if isinstance(text, str) and image is None: # text only embedding case
                conversation = text_only_input(text)
            elif isinstance(image, str) and text is None:
                conversation = image_only_input(image)
            elif isinstance(image, str) and isinstance(text, str):
                conversation = image_and_inst_input(text, image)
                use_adapter = True
            else:
                raise Exception("NotSupportedModalityError")
            
            text_input = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(conversation)

            inps = processor(
                text=text_input,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            inps = _prepare_input(inps)
            output = model.inst_embed(inps, not use_adapter)
            
            return output
        return functools.partial(embed, model, processor)

def get_batched_embed_function(base_model, pretrain_adapter, instruction_adapter):
        min_pixels = 256*28*28
        max_pixels = 1024*28*28
        from transformers import AutoProcessor
        
        model = get_model(base_model, pretrain_adapter, instruction_adapter)

        processor = AutoProcessor.from_pretrained(base_model,
                                                padding_side="right",
                                                use_fast=False,
                                                max_pixels=max_pixels,
                                                min_pixels=min_pixels)
        
    
        def embed(model, processor, image = None, text = None):

            conversation = None
            use_adapter = False
            
            if isinstance(text, list) and image is None: # text only embedding case
                conversation = [text_only_input(t) for t in text]
            elif isinstance(image, list) and text is None:
                conversation = [image_only_input(i) for i in image]
            elif isinstance(image, list) and isinstance(text, list):
                conversation = [image_and_inst_input(x, y) for (x,y) in zip(text,image)]
                use_adapter = True
            else:
                raise Exception("NotSupportedModalityError")
            
            text_input = [processor.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True
            ) for c in conversation]
            image_inputs, _ = process_vision_info(conversation)

            inps = processor(
                text=text_input,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            inps = _prepare_input(inps)
            output = model.inst_embed(inps, not use_adapter)
            
            return output
        return functools.partial(embed, model, processor)