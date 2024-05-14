from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm2vec import LLM2Vec


class APDModel(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        vision_enc: nn.Module,
        text_enc: LLM2Vec,
        text_enc_output_dim: int,
        text_proj_sizes: List[int],
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.vision_enc = vision_enc
        self.lock_image_tower()

        self.text_enc = text_enc
        # freeze text tower
        for param in self.text_enc.parameters():
            param.requires_grad = False

        # create projector for pretrained text encoder to match output dim and
        assert len(text_proj_sizes) > 0 
        assert self.vision_enc.output_dim == text_proj_sizes[-1], \
            "Output dim of text projector != Output dim of vision encoder"
        self.text_proj = nn.ModuleList(
            [nn.Linear(text_enc_output_dim, text_proj_sizes[0], bias=False)]
        )
        for k in range(1, len(text_proj_sizes)):
            self.text_proj.append(nn.ReLU())
            self.text_proj.append(nn.Linear(text_proj_sizes[k-1], text_proj_sizes[k], bias=False))

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.vision_enc.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.vision_enc.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.vision_enc(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False, device: str = "cuda"):
        x = self.text_enc.encode(text, show_progress_bar=False, device=device)  # returns embeddings on cpu
        x.to(device)
        for layer in self.text_proj:
            x = layer(x)
        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
    ):
        device = image.device
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True, device=device) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()
