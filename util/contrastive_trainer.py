from typing import List, Optional, Tuple, Dict, Union, Any
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import has_length, EvalLoopOutput
from transformers.trainer import RandomSampler, logger
import torch.distributed as dist
from transformers.integrations import WandbCallback, deepspeed_init
import os
import time

class WandbLogger(WandbCallback):

    def __init__(self):
        super().__init__()
        self.additional_metrics = {}

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        if state.is_world_process_zero:
            self._wandb.config.update({"pid": str(os.getpid())}, allow_val_change=True)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model=model, logs=logs | self.additional_metrics, **kwargs)
        self.additional_metrics.clear()

class ContrastiveTrainer(Trainer):

     def __init__(self, wandb=True, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.enable_wandb=wandb
          if self.enable_wandb:
               self.wandb_callback = WandbLogger()
               self.add_callback(self.wandb_callback)

     def log_to_wandb(self, key, value):
          if self.enable_wandb:
               self.wandb_callback.additional_metrics[key] = value

     def prediction_step(
          self,
          model: torch.nn.Module,
          inputs: Dict[str, Union[torch.Tensor, Any]],
          prediction_loss_only: bool,
          ignore_keys: Optional[List[str]] = None,
     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

          inputs = self._prepare_inputs(inputs)

          with torch.no_grad():
               with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
               loss = loss.mean().detach()

          return loss, outputs
     
     def embed_step(
          self,
          model: torch.nn.Module,
          inputs: Dict[str, Union[torch.Tensor, Any]],
          prediction_loss_only: bool,
          ignore_keys: Optional[List[str]] = None,
     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

          inputs = self._prepare_inputs(inputs)

          with torch.no_grad():
               with self.compute_loss_context_manager():
                    out = model(inputs, return_outputs=True, return_prediction=True)
          loss, outputs = out
          return loss, outputs

     def evaluation_loop(
          self,
          dataloader: torch.utils.data.DataLoader,
          description: str,
          prediction_loss_only: Optional[bool] = None,
          ignore_keys: Optional[List[str]] = None,
          metric_key_prefix: str = "eval",
     ) -> EvalLoopOutput:
          """
          Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

          Works both with or without labels.
          """
          args = self.args

          prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

          # if eval is called w/o train, handle model prep here
          if self.is_deepspeed_enabled and self.deepspeed is None:
               _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

          model = self._wrap_model(self.model, training=False, dataloader=dataloader)

          if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

          # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
          # while ``train`` is running, cast it to the right dtype first and then put on device
          if not self.is_in_train:
               if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
               elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)

          batch_size = self.args.eval_batch_size

          logger.info(f"\n***** Running {description} *****")
          if has_length(dataloader):
               logger.info(f"  Num examples = {self.num_examples(dataloader)}")
          else:
               logger.info("  Num examples: Unknown")
          logger.info(f"  Batch size = {batch_size}")

          model.eval()

          self.callback_handler.eval_dataloader = dataloader

          if args.past_index >= 0:
               self._past = None

          # Will be useful when we have an iterable dataset so don't know its length.
          observed_num_examples = 0
          outputs = []
          losses = []
          predictions = []

          # Main evaluation loop
          for _, inputs in enumerate(dataloader):
               # Update the observed num examples
               observed_batch_size = find_batch_size(inputs)
               if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                         batch_size = observed_batch_size

               # Prediction step
               loss, output = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
               if "prediction" in output.keys():
                    predictions.append(output.pop("prediction"))
               outputs.append(output)
               losses.append(loss)
               
               self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)


          # Number of samples
          num_samples = observed_num_examples

          metrics = {}
     
     # Iterate through each dictionary in the list
          for d in outputs:
               for key, value in d.items():
                    metrics[key] = metrics.get(key, 0) + value
     
          metrics = {key: metrics[key] / len(outputs) for key in metrics}
          metrics["loss"] = torch.mean(torch.stack(losses, dim=0),dim=0)
          metrics=cast_loss_dict(metrics, metric_key_prefix)

          return EvalLoopOutput(predictions=predictions, label_ids=None, metrics=metrics, num_samples=num_samples)

     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

          loss, outputs = model(inputs, return_outputs=True)
          if not return_outputs: self.log_output(dict(outputs, **{"batch_loss": loss}))

          return (loss, outputs) if return_outputs else loss

     def log_output(self, outputs):
          
          for k, v in outputs.items():
               if isinstance(v, torch.Tensor):
                    v = v.detach()
                    dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    v = v / dist.get_world_size()

               if dist.get_rank() == 0:
                    self.log_to_wandb(k,v)

     def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
          return RandomSampler(self.train_dataset)
     

def cast_loss_dict(d: Dict, dataset_name: str):
     return {dataset_name+"_"+x:y.cpu().item() for (x,y) in d.items()}
