import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Callable, Optional, Tuple

from stepnav.models.field2prior import Field2Prior

class Vjepa2(nn.Module):
    def __init__(self, encoder = 'vjepa2_large', encoding_size = 256, len_traj_pred = 8):
        super().__init__()
        self.imsize = 256
        self.encoding_size = encoding_size
        if encoder == 'vjepa2_large':
            self.encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
            self.vjepa_size = 1024 * 2
        elif encoder == 'vjepa2_huge':
            self.encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
            self.vjepa_size = 1280 * 2
        elif encoder == 'vjepa2_giant':
            self.encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant_384')
            self.vjepa_size = 1408 * 2
        else:
            raise ValueError("Unsupported encoder type. Use 'vjepa2_large', 'vjepa2_huge' or 'vjepa2_giant'.")
        
        self.fc = nn.Linear(self.vjepa_size, self.encoding_size)
        
        self.field2prior = Field2Prior(context_dim=self.encoding_size, num_final_trajectories=1, trajectory_length=len_traj_pred)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def preprocess(self, x):
        """
            x: [B, T, C, H, W]  uint8(0..255) 或 float(0..1)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        B, T, C, H, W = x.shape

        if (H, W) != (self.imsize, self.imsize):
            x = x.view(B*T, C, H, W)
            x = F.interpolate(x, size=(self.imsize, self.imsize), mode='bicubic', align_corners=False)
            x = x.view(B, T, C, self.imsize, self.imsize)
            
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,1,3,1,1)
        
        x = (x - mean) / std
        
        x_bcthw = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        return x_bcthw
    
    def forward(self,
                obs_img: torch.tensor,
                goal_img: torch.tensor,
                input_goal_mask: torch.tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_img: [B, T*C, H, W]  observation images
        goal_img: [B, C, H, W]  goal images
        input_goal_mask: [B, T]  goal mask (optional)
        """
        device = obs_img.device
        
        B, C, H, W = goal_img.shape
        B, TC, H, W = obs_img.shape
        
        # Get the input goal mask
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # Reshape
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0) # [B*T, C, H, W]
        obs_img = obs_img.reshape(B, TC//C, C, H, W)  # [B, T, C, H, W]
        
        goal_img = goal_img.unsqueeze(1).repeat(1, TC//C, 1, 1, 1)  # [B, T, C, H, W]
        
        
        # Preprocess images & encode
        obs_img = self.preprocess(obs_img)  # [B, C, T, H, W]
        obs_encoding = self.encoder(obs_img)  # [B, C, T, H, W] -> [B, 512, 1408]

        goal_img = self.preprocess(goal_img)  # [B, C, T, H, W]
        goal_encoding = self.encoder(goal_img)  # [B, C, T, H, W] -> [B, 512, 1408]

        feat = torch.concat([obs_encoding, goal_encoding], dim=2)  # [B, 512, 1408*2]
        feat = feat.mean(dim=1) # [B, 1408 * 2]
        feat = self.fc(feat)  # [B, 1408 * 2] -> [B, encoding_size]
        
        traj_prior = self.field2prior(feat)
        
        return feat, traj_prior
    
    
if __name__ == "__main__":
    model = Vjepa2(encoder='vjepa2_huge')
    obs_img = torch.randn(2, 9, 256, 256)  # Example observation images
    goal_img = torch.randn(2, 3, 256, 256)  # Example goal image
    output_feat, output_traj_prior = model(obs_img, goal_img)
    print(output_feat.shape)  # Should print the shape of the output tensor
    print(output_traj_prior.shape)  # Should print the shape of the output tensor