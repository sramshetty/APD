import time
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from open_clip.loss import ClipLoss, SigLipLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class APDTrainer:

    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model
    
    def train(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataloader: torch.utils.data.DataLoader,
        loss_type: Optional[str] = "clip",
        num_epochs: Optional[int] = 5,
        log_steps: Optional[int] = 1000,
        save_steps: Optional[int] = 10000,
        save_path: Optional[str] = "./apd.pt",
        use_wandb: Optional[bool] = False,
        run_name: Optional[str] = "",
        device: Optional[str] = "cuda",
    ):  
        self.model.train()
        self.model.to(device)

        criterion = ClipLoss() if loss_type == "clip" else SigLipLoss()

        num_batches_per_epoch = len(dataloader)
        total_steps = num_epochs * num_batches_per_epoch

        if use_wandb:
            wandb.login()
            wandb.init(
                project="aligning_pretrained_decoders",
                name=run_name,
                config={
                    "projection_dims": self.model.text_proj_sizes,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epochs": num_epochs,
                    "loss_type": loss_type,
                },
            )

        for ep in range(num_epochs):

            losses_m = {}
            batch_time_m = AverageMeter()
            data_time_m = AverageMeter()
            end = time.time()

            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {ep}")):
                curr_step = (ep * num_batches_per_epoch) + i

                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                
                data_time = time.time() - end
                data_time_m.update(data_time)

                optimizer.zero_grad()

                model_out = self.model(images, texts)
                logit_scale = model_out["logit_scale"]
                losses = criterion(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

                total_loss.backward()

                optimizer.step()
                scheduler.step()

                batch_time = time.time() - end
                batch_time_m.update(batch_time)
                end = time.time()

                if use_wandb:
                    log = {
                        "logit_scale": logit_scale.item(),
                        "data_time": data_time,
                        "batch_time": batch_time,
                    }
                    log.update(
                        {
                            loss_name.capitalize(): loss_val 
                            for loss_name, loss_val in losses.items()
                        }
                    )
                    wandb.log(log)

                if curr_step % log_steps == 0 or i == num_batches_per_epoch - 1:
                    batch_size = len(images)
                    for key, val in losses.items():
                        if key not in losses_m:
                            losses_m[key] = AverageMeter()
                        losses_m[key].update(val.item(), batch_size)
                    
                    loss_log = "|".join(
                        [
                            f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                            for loss_name, loss_m in losses_m.items()
                        ]
                    )
                    
                    print(f"Losses at step {curr_step}/{total_steps}:\n{loss_log}")
                    print(f"Data Time: {data_time_m.val}")
                    print(f"Batch Time: {batch_time_m.val}")

                    batch_time_m.reset()
                    data_time_m.reset()
            
                if curr_step % save_steps == 0 or i == num_batches_per_epoch - 1:
                    checkpoint_dict = {
                        "step": curr_step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(checkpoint_dict, save_path)
        
        if use_wandb:
            wandb.finish()
