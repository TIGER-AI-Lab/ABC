from dataclasses import dataclass, field
from typing import Any, Optional
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    adapter: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA adapter to load.'}
    )
    merge: Optional[str] = field(
        default=False,
        metadata={'help': 'LoRA where to merge adapter into underlying model.'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )

    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )

    lora_dropout: float = field(
        default=0.05,
        metadata={'help': 'Dropout rate for LORA layers'}
    )

    model_architecture: str = field(
        default=None,
        metadata={'help': 'The architecture (model defintion)'}
    )

    instruction_mode: str = field(
        default=False,
        metadata={'help': 'Whether pretraining or instruction finetuning'}
    )

    use_dora: str = field(
        default=False,
        metadata={'help': 'Whether to use dora as the adapter'}
    )

@dataclass
class VLMTrainingArguments(TrainingArguments):
    loss_type: Optional[str] = field(
        default=None,
        metadata={"help": "Loss type used in ContrastiveTrainer"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """

    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )

    negatives: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of negatives used for each query'
            
        },
    )

    force_image_size: Optional[int] = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )

    eval_datasets: Optional[Any] = field(
            default_factory=lambda: [],
            metadata={'help': 'List of datasets to eval against'},
    )

    eval_steps_per_dataset: Optional[int] = field(
            default=1,
            metadata={'help': 'Number of steps to use each time model is evaluated'},
    )
    training_dataset_name: Optional[str] = field(
            default=None,
            metadata={'help': 'Name of datasets to use in training'},
    )