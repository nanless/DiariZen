# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import torch
import numpy as np

from accelerate.logging import get_logger

from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.loss import binary_cross_entropy

from diarizen.trainer_single_opt import Trainer as BaseTrainer

logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator.print(self.model)

        # auto GN
        self.grad_history = []

    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 

    def auto_clip_grad_norm_(self, model):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        if len(self.grad_history) > self.gradient_history_size:
            self.grad_history.pop(0)
        clip_value = np.percentile(self.grad_history, self.gradient_percentile)
        self.accelerator.clip_grad_norm_(model.parameters(), clip_value)  

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        xs, target = batch['xs'], batch['ts'] 
        frame_mask = batch.get('mask', None)
        if frame_mask is not None:
            frame_mask = frame_mask.to(xs.device).unsqueeze(-1)  # (B, T, 1)

        y_pred = self.model(xs)
        # align sequence lengths between prediction and target
        min_len = min(y_pred.size(1), target.size(1))
        if y_pred.size(1) != min_len:
            y_pred = y_pred[:, :min_len, :].contiguous()
        if target.size(1) != min_len:
            target = target[:, :min_len, :]
            if frame_mask is not None:
                frame_mask = frame_mask[:, :min_len, :]
        
        # 使用原生 multilabel，不需要 powerset 转换
        # y_pred 已经是 sigmoid 输出，形状为 (B, T, num_speakers)
        # target 形状为 (B, T, num_speakers)
        permutated_target, _ = permutate(y_pred, target)
       
        loss = binary_cross_entropy(
            y_pred,
            permutated_target,
            weight=frame_mask
        )
        
        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None
        
        self.accelerator.backward(loss)

        grad_norm = None
        if self.accelerator.sync_gradients:
            # The gradients are added across all processes in this cumulative gradient accumulation step.
            grad_norm = self.compute_grad_norm(self.model)
            self.auto_clip_grad_norm_(self.model)
                               
        self.optimizer.step()
        
        return {
            "Loss": loss.detach(),
            "grad_norm": grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        xs, target = batch['xs'], batch['ts'] 
        frame_mask = batch.get('mask', None)
        if frame_mask is not None:
            frame_mask = frame_mask.to(xs.device).unsqueeze(-1)  # (B, T, 1)
        sil_all_target = torch.zeros_like(target)

        y_pred = self.model(xs)
        # align sequence lengths between prediction and target
        min_len = min(y_pred.size(1), target.size(1))
        if y_pred.size(1) != min_len:
            y_pred = y_pred[:, :min_len, :].contiguous()
        if target.size(1) != min_len:
            target = target[:, :min_len, :]
            if frame_mask is not None:
                frame_mask = frame_mask[:, :min_len, :]
        
        # 使用原生 multilabel，不需要 powerset 转换
        permutated_target, _ = permutate(y_pred, target)

        loss = binary_cross_entropy(
            y_pred,
            permutated_target,
            weight=frame_mask
        )
        
        # 使用排列对齐后的目标计算指标
        if frame_mask is not None:
            mask_metric = frame_mask.transpose(1, 2)  # (B, 1, T)
            val_metrics = self.unwrap_model.validation_metric(
                torch.transpose(y_pred, 1, 2) * mask_metric,
                torch.transpose(permutated_target, 1, 2) * mask_metric,
            )
        else:
            val_metrics = self.unwrap_model.validation_metric(
                torch.transpose(y_pred, 1, 2),
                torch.transpose(permutated_target, 1, 2),
            )

        # 检查是否有语音（使用排列对齐后的目标）
        if not torch.equal(permutated_target, sil_all_target):
            val_DER = val_metrics['DiarizationErrorRate']
            val_FA = val_metrics['DiarizationErrorRate/FalseAlarm']
            val_Miss = val_metrics['DiarizationErrorRate/Miss']
            val_Confusion = val_metrics['DiarizationErrorRate/Confusion']
        else:
            val_DER = torch.zeros_like(val_metrics['DiarizationErrorRate'])
            val_FA = torch.zeros_like(val_metrics['DiarizationErrorRate/FalseAlarm'])
            val_Miss = torch.zeros_like(val_metrics['DiarizationErrorRate/Miss'])
            val_Confusion = torch.zeros_like(val_metrics['DiarizationErrorRate/Confusion'])

        return {"Loss": loss, "DER": val_DER, "FA": val_FA, "Miss": val_Miss, "Confusion": val_Confusion}

    def validation_epoch_end(self, validation_epoch_output):
        metric_keys = validation_epoch_output[0].keys()
        # Compute mean loss on all loss items on a epoch
        for key in metric_keys:
            metric_items = [torch.mean(step_out[key]) for step_out in validation_epoch_output]
            metric_mean = torch.mean(torch.tensor(metric_items))
            if key == "Loss":
                Loss_val = metric_mean
            if key == "DER":
                DER_val = metric_mean
            self.writer.add_scalar(f"Validation_Epoch/{key}", metric_mean, self.state.epochs_trained)
        logger.info(f"Validation Loss/DER on epoch {self.state.epochs_trained}: {round(Loss_val.item(), 3)} / {round(DER_val.item(), 3)}")
        # metric reset
        self.unwrap_model.validation_metric.reset()
        return Loss_val

    def train(self, train_dataloader, validation_dataloader):
        """Override train method to add progress bar updates with loss, grad_norm, and lr."""
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        
        early_stop_mark = torch.zeros(1, device=self.device)
        
        # Setting up training control variables
        steps_per_epoch = len(train_dataloader)
        update_steps_per_epoch = steps_per_epoch // self.gradient_accumulation_steps
        update_steps_per_epoch = max(update_steps_per_epoch, 1)
        
        if self.max_steps > 0:
            max_steps = self.max_steps
            max_epochs = self.max_steps // update_steps_per_epoch + int(self.max_steps % update_steps_per_epoch > 0)
        else:
            max_steps = self.max_epochs * update_steps_per_epoch
            max_epochs = self.max_epochs
        
        logger.info("Training control variables:")
        logger.info(f"`steps_per_epoch`: {steps_per_epoch}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"`update_steps_per_epoch`: {update_steps_per_epoch}")
        logger.info(f"`max_steps`: {max_steps}")
        logger.info(f"`max_epochs`: {max_epochs}")
        
        # Generator learning rate scheduler
        if self.warmup_steps > 0:
            self.create_schedulers(max_steps=max_steps)
        if self.use_one_cycle_lr:
            self.create_lr_one_cycle_scheduler(max_steps=max_steps * self.accelerator.num_processes)
        
        # Resume
        if self.resume:
            self._load_checkpoint(ckpt_path="latest")
        
        # validation 0 epoch performance
        if self.validation_before_training:
            with torch.no_grad():
                logger.info("Validation on ZERO epoch...")
                score = self.validate(validation_dataloader)
            if self.accelerator.is_local_main_process:
                self._save_checkpoint(epoch=0, is_best_epoch=False)
        
        for epoch in range(self.state.epochs_trained + 1, max_epochs + 1):
            logger.info(f"{'=' * 9} Epoch {epoch} out of {max_epochs} {'=' * 9}")
            logger.info("Begin training...")
            
            self.set_models_to_train_mode()
            if self.freeze_wavlm:
                self.unwrap_model.wavlm_model.eval()
            
            training_epoch_output = []
            
            dataloader_bar = tqdm(
                train_dataloader,
                desc="",
                dynamic_ncols=True,
                bar_format="{l_bar}{r_bar}",
                colour="green",
                disable=not self.accelerator.is_local_main_process,
                position=0,
                leave=True,
            )
            
            for batch_idx, batch in enumerate(dataloader_bar):
                with self.accelerator.accumulate(self.model):
                    loss_dict = self.training_step(batch, batch_idx)
                    training_epoch_output.append(loss_dict)
                    
                    if not self.accelerator.optimizer_step_was_skipped:
                        if self.warmup_steps > 0:
                            self.lr_scheduler_step()
                        
                        if self.use_one_cycle_lr:
                            self.lr_one_cycle_scheduler.step()
                
                # Update progress bar with loss, grad_norm, and lr
                if self.accelerator.is_local_main_process and loss_dict is not None:
                    loss_val = loss_dict.get("Loss", None)
                    grad_norm = loss_dict.get("grad_norm", None)
                    lr = loss_dict.get("lr", None)
                    desc_parts = []
                    if loss_val is not None:
                        try:
                            desc_parts.append(f"loss {float(loss_val):.4f}")
                        except Exception:
                            pass
                    if grad_norm is not None:
                        try:
                            desc_parts.append(f"gn {float(grad_norm):.1f}")
                        except Exception:
                            pass
                    if lr is not None:
                        try:
                            desc_parts.append(f"lr {float(lr):.2e}")
                        except Exception:
                            pass
                    if desc_parts:
                        dataloader_bar.set_description(" | ".join(desc_parts))
                
                self.state.steps_trained += 1
            self.state.epochs_trained += 1
            self.training_epoch_end(training_epoch_output)
            
            # Should save, evaluate, and early stop?
            if self.accelerator.is_local_main_process and epoch % self.save_ckpt_interval == 0:
                self._save_checkpoint(epoch, is_best_epoch=False)
            
            if epoch % self.validation_interval == 0:
                with torch.no_grad():
                    logger.info("Training finished, begin validation...")
                    score = self.validate(validation_dataloader)
                    
                    if self.accelerator.is_local_main_process:
                        if self.lr_decay:
                            self.lr_decay_scheduler.step(score)
                        
                        should_stop = self._run_early_stop_check(score)
                        if should_stop:
                            early_stop_mark += 1
                    
                    logger.info("Validation finished.")
            
            self.accelerator.wait_for_everyone()
            
            reduced_early_stop_mark = self.accelerator.reduce(early_stop_mark, reduction="sum")
            
            if reduced_early_stop_mark != 0:
                break

