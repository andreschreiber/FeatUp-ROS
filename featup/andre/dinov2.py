import torch
import torch.nn as nn
from featup.layers import ChannelNorm
from featup.upsamplers import get_upsampler


class DINOv2Featurizer(nn.Module):
    def __init__(self, arch, patch_size, feat_type):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type
        self.n_feats = 128
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def forward(self, img):
        h = img.shape[2] // self.patch_size
        w = img.shape[3] // self.patch_size
        feats = self.model.forward_features(img)
        return {
            "patch_tokens": feats["x_norm_patchtokens"].reshape(-1, h, w, 384).permute(0, 3, 1, 2),
            "cls_token": feats["x_norm_clstoken"]
        }

    @staticmethod
    def get_featurizer(activation_type="key", **kwargs):
        patch_size = 14
        dim = 384
        model = DINOv2Featurizer("dinov2_vits14", patch_size, activation_type)
        return model, patch_size, dim


class DINOv2UpFeatBackbone(nn.Module):
    def __init__(self, use_norm):
        super().__init__()
        model, patch_size, self.dim = DINOv2Featurizer.get_featurizer("token", num_classes=1000)
        self.use_norm = use_norm
        if use_norm:
            self.model = torch.nn.Sequential(model, ChannelNorm(self.dim))
        else:
            self.model = model
        self.upsampler = get_upsampler("jbu_stack", self.dim)

    def forward(self, image):
        if self.use_norm:
            model_out = self.model[0](image)
            model_out['patch_tokens'] = self.model[1](model_out['patch_tokens'])
        else:
            model_out = self.model(image)
        
        return {
            "features": self.upsampler(model_out['patch_tokens'], image), 
            "cls_token": model_out["cls_token"]
        }

    @staticmethod
    def load_backbone(pretrained=True, use_norm=True):
        model_name = "dinov2"
        model = DINOv2UpFeatBackbone(use_norm)
        if pretrained:
            if use_norm:
                exp_dir = ""
            else:
                exp_dir = "no_norm/"
            checkpoint_url = f"https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/{exp_dir}{model_name}_jbu_stack_cocostuff.ckpt"
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if "scale_net" not in k and "downsampler" not in k}
            model.load_state_dict(state_dict, strict=False)
        return model
