import torch
import torch.distributed as dist
import os, sys, logging
from transformers import set_seed, HfArgumentParser
from util.contrastive_trainer import ContrastiveTrainer
from util.dataclass import ModelArguments, DataTrainingArguments, VLMTrainingArguments
from transformers import AutoProcessor
from qwen.qwen_dataset import build_contrastive_dataset, build_eval_datasets, QwenCollate
from model.modeling_abc import MODEL_ARCHITECTURE
from monkey_patch.qwen_attn_patch import monkey_patch_transformers_lib, unmask_attn_monkey_patch
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def load_model(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: VLMTrainingArguments):

    min_pixels = data_args.min_dynamic_patch*28*28
    max_pixels = data_args.max_dynamic_patch*28*28
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                                padding_side="right",
                                                use_fast=False,
                                                max_pixels=max_pixels,
                                                min_pixels=min_pixels)

    model = MODEL_ARCHITECTURE[model_args.model_architecture].from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    model.instruction_mode = model_args.instruction_mode

    if model_args.adapter:
        model = PeftModel.from_pretrained(model, model_args.adapter, is_trainable=True)
        if model_args.merge:
            model = model.merge_and_unload()

    return model, processor

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))

    monkey_patch_transformers_lib()

    if MODEL_ARCHITECTURE[model_args.model_architecture].attn_mask == 'bidirectional':
        unmask_attn_monkey_patch()
    elif MODEL_ARCHITECTURE[model_args.model_architecture].attn_mask != 'casual':
        raise Exception("NotImplementedError")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model, tokenizer = load_model(model_args, data_args, training_args)

    train_dataset = build_contrastive_dataset(
    data_args,
    tokenizer,
    dataset_name=data_args.training_dataset_name,
    is_train=True
    )  

    eval_dataset = build_eval_datasets(
        training_args.per_device_eval_batch_size,
        data_args,
        tokenizer
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_params(module):
        for param in module.parameters():
            param.requires_grad = True

    _freeze_params(model)

    # Freeze base model weights
    if model_args.instruction_mode:
        if dist.get_rank() == 0: print("Mode: instruction finetuning")
        # Only train temperature when instruction finetuning
        modules_to_save = None
    else:
        if dist.get_rank() == 0: print("Mode: pretraining")
        _unfreeze_params(model.mlp_head)
        _unfreeze_params(model.temperature)
        # train and save both temperature and mlp_head
        modules_to_save = ["temperature","mlp_head"]

    if model_args.grad_checkpoint:
        model.model.gradient_checkpointing_enable()
        model.visual.gradient_checkpointing_enable()

    target_modules = []

    # LoRA for LLM
    if model_args.use_llm_lora:
        target_modules.extend(['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'])

    # LoRA for vision backbone
    if model_args.use_backbone_lora:
        target_modules.extend(['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'])
        
    if len(target_modules):
        lora_config = LoraConfig(
            r=model_args.use_llm_lora,
            target_modules=target_modules,
            lora_alpha=2*model_args.use_llm_lora,
            lora_dropout=model_args.lora_dropout,
            use_dora=model_args.use_dora,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, lora_config)

        # set as a function so that it is not tracked by autograd
        setattr(model.get_base_model(), "get_peft_wrapper", lambda: model)
        model.print_trainable_parameters()

    # print trainable parameters
    if dist.get_rank() == 0:
        print("PARAMS BEING TRAINED:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=QwenCollate(tokenizer)
    )

    # Training
    if training_args.do_train:
        checkpoint = None

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()