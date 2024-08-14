import torch

# modify as desired to download what you want (meant to be used in Dockerfile)
featup = torch.hub.load('mhamilton723/FeatUp', 'dinov2', use_norm=True)