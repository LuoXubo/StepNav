from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import click
from depth_anything_v2.depth_anything_v2.dinov2 import DINOv2
from stepnav.models.attention import PositionalEncoding, DIFP


class VJEPA2NavigationEncoder(nn.Module):
    """V-JEPA encoder wrapper for navigation tasks"""
    def __init__(
        self,
        model_name='vjepa2_vit_large', 
        img_size=224,          
        normalize_features=True,
        device='cuda',
    ):
        super().__init__()
        click.echo(
            click.style(f">> Loading {model_name} from torch hub on device {device}...", fg="green")
        )
        
        try:
            self.vjepa_model = torch.hub.load('facebookresearch/vjepa2', model_name, pretrained=True)
        except Exception as e:
            click.echo(
                click.style(f">> Error loading from torch hub: {e}", fg="red")
            )
            raise e

        self.img_size = img_size
        self.device = device
        self.normalize_features = normalize_features
        
        # Unpack model (Encoder / Predictor)
        if isinstance(self.vjepa_model, tuple):
            self.encoder, self.predictor = self.vjepa_model
        else:
            self.encoder = self.vjepa_model
            self.predictor = None
            
        self.encoder.to(device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        click.echo(
            click.style(f">> Successfully loaded V-JEPA model \"{model_name}\".", fg="green")
        )
            
        # Get embedding dimension with dummy input (T=2 for temporal kernel)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 2, img_size, img_size).to(device)
            out = self.encoder(dummy)
            self.embed_dim = out.shape[-1]

    def fast_batch_preprocess(self, images_tensor, is_stacked=True):
        """
        Preprocess images for V-JEPA
        Args:
            images_tensor: [B, C, H, W] where C = T*3 if stacked, else C = 3
            is_stacked: whether input contains multiple frames stacked in channel dimension
        Returns:
            [B, 3, T, H, W] format for V-JEPA
        """
        B, C_in, H, W = images_tensor.shape
        C_per_frame = 3
        
        # 1. Reshape to separate frames
        if is_stacked:
            T = C_in // C_per_frame 
            x = images_tensor.view(B, T, C_per_frame, H, W).reshape(B * T, C_per_frame, H, W)
        else:
            T = 1
            x = images_tensor 

        # 2. Resize to V-JEPA input size
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # 3. ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # 4. Reshape to V-JEPA format [B, 3, T, H, W]
        x = x.view(B, T, C_per_frame, self.img_size, self.img_size).permute(0, 2, 1, 3, 4)
        
        # 5. Handle single frame case (temporal kernel requires T >= 2)
        if x.shape[2] < 2:
            x = x.repeat(1, 1, 2, 1, 1)
            
        return x

    def forward(self, x):
        """
        Args:
            x: [B, 3, T, H, W]
        Returns:
            [B, D] global features
        """
        with torch.no_grad():
            features = self.encoder(x)  # [B, N, D]
        
        if self.normalize_features:
            features = F.layer_norm(features, (features.size(-1),))
        
        # Global average pooling over spatial tokens
        return features.mean(dim=1)  # [B, D]


class NoMaD_VJEPA(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        vjepa_model_name: str = 'vjepa2_vit_large',
        vjepa_img_size: Optional[int] = 224,
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        depth_cfg: Optional[dict] = {},
        use_depth: bool = True,
        device: str = 'cuda',
    ) -> None:
        """
        NoMaD ViNT Encoder with V-JEPA
        Args:
            context_size: number of historical observation frames
            vjepa_model_name: V-JEPA model variant
            vjepa_img_size: input image size for V-JEPA
            obs_encoding_size: size of compressed observation encoding
            use_depth: whether to use depth encoder
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.depth_cfg = depth_cfg
        self.use_depth = use_depth

        # Initialize V-JEPA observation encoder
        click.echo(click.style(">> Initializing V-JEPA observation encoder...", fg="cyan"))
        self.obs_encoder = VJEPA2NavigationEncoder(
            model_name=vjepa_model_name,
            img_size=vjepa_img_size,
            normalize_features=True,
            device=device,
        )
        self.num_obs_features = self.obs_encoder.embed_dim
        
        # Initialize V-JEPA goal encoder (shares architecture with obs encoder)
        click.echo(click.style(">> Initializing V-JEPA goal encoder...", fg="cyan"))
        self.goal_encoder = VJEPA2NavigationEncoder(
            model_name=vjepa_model_name,
            img_size=vjepa_img_size,
            normalize_features=True,
            device=device,
        )
        self.num_goal_features = self.goal_encoder.embed_dim

        # Initialize compression layers
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size
            )
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(
                self.num_goal_features, self.goal_encoding_size
            )
        else:
            self.compress_goal_enc = nn.Identity()

        # Initialize Depth Encoder (optional)
        if self.use_depth:
            click.echo(click.style(">> Initializing depth encoder...", fg="cyan"))
            self.depth_enc_str = depth_cfg["depth_encoder"]
            self.depth_encoder = DINOv2(model_name=self.depth_enc_str)
            for param in self.depth_encoder.parameters():
                param.requires_grad = False
            self.depth_layer_idx = depth_cfg["dino_layer_idx"][self.depth_enc_str]
            self.depth_pool_dim = depth_cfg["pool_dim"]
            self.depth_enc_dim = depth_cfg["out_dim"][self.depth_enc_str]
            self.num_depth_features = self.depth_enc_dim * self.depth_pool_dim
            
            if self.num_depth_features != self.goal_encoding_size:
                self.compress_depth_enc = nn.Sequential(
                    nn.AdaptiveAvgPool1d(self.depth_pool_dim),
                    nn.Flatten(),
                    nn.Linear(self.num_depth_features, self.goal_encoding_size),
                )
            else:
                self.compress_depth_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        num_tokens = self.context_size + 2  # obs tokens + goal token
        if self.use_depth:
            num_tokens += 1  # + depth token
            
        self.positional_encoding = PositionalEncoding(
            self.obs_encoding_size, max_seq_len=num_tokens
        )
        
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(
            self.sa_layer, num_layers=mha_num_attention_layers
        )

        # Goal mask definition (0 = no mask, 1 = mask)
        mask_size = num_tokens
        self.goal_mask = torch.zeros((1, mask_size), dtype=torch.bool)
        self.goal_mask[:, -1] = True  # Mask out the goal token
        self.no_mask = torch.zeros((1, mask_size), dtype=torch.bool)
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        
        # Average pooling mask
        self.avg_pool_mask = torch.cat(
            [
                1 - self.no_mask.float(),
                (1 - self.goal_mask.float()) * ((num_tokens - 1) / (num_tokens - 2)),
            ],
            dim=0,
        )

        # DIFP: Dynamics-Inspired Feature Refinement module
        # Refines the temporal feature sequence Z via a goal-conditional
        # variational formulation to produce smooth, goal-aligned features.
        self.difp = DIFP(
            feature_dim=self.obs_encoding_size,
            max_seq_len=num_tokens,
        )

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_img: [B, C, H, W] where C = 3*(context_size+1)
            goal_img: [B, 3, H, W]
            input_goal_mask: [B] binary mask (1 = no goal available)
        Returns:
            obs_encoding_tokens: [B, obs_encoding_size]
        """
        device = obs_img.device
        batch_size = obs_img.size(0)

        # ============ Process Observation Images with V-JEPA ============
        # obs_img contains context_size+1 frames stacked in channel dimension
        # Split into individual frames and process each with V-JEPA
        obs_img_split = torch.split(obs_img, 3, dim=1)  # List of [B, 3, H, W]
        obs_encodings_list = []
        
        for single_obs in obs_img_split:
            # Preprocess for V-JEPA (will duplicate frame since T=1)
            vjepa_input = self.obs_encoder.fast_batch_preprocess(single_obs, is_stacked=False)
            # Encode
            obs_enc = self.obs_encoder(vjepa_input)  # [B, D]
            obs_encodings_list.append(obs_enc)
        
        # Stack all observation encodings
        obs_encoding = torch.stack(obs_encodings_list, dim=1)  # [B, context_size+1, D]
        obs_encoding = self.compress_obs_enc(obs_encoding)  # [B, context_size+1, obs_encoding_size]

        # ============ Process Goal Image with V-JEPA ============
        # Concatenate current observation with goal for context
        current_obs = obs_img[:, 3 * self.context_size:, :, :]  # [B, 3, H, W]
        obsgoal_img = torch.cat([current_obs, goal_img], dim=1)  # [B, 6, H, W]
        
        # Preprocess for V-JEPA (2 frames: current obs + goal)
        vjepa_goal_input = self.goal_encoder.fast_batch_preprocess(obsgoal_img, is_stacked=True)
        goal_encoding = self.goal_encoder(vjepa_goal_input)  # [B, D]
        goal_encoding = self.compress_goal_enc(goal_encoding).unsqueeze(1)  # [B, 1, goal_encoding_size]

        # ============ Process Depth (if enabled) ============
        if self.use_depth:
            depth_inp = current_obs
            depth_inp = F.pad(depth_inp, (1, 1, 1, 1), mode="constant", value=0)
            dpt_enc_all = self.depth_encoder.get_intermediate_layers(
                depth_inp, self.depth_layer_idx, return_class_token=False
            )
            dpt_enc_last = dpt_enc_all[-1].permute(0, 2, 1)  # [B, C, N]
            depth_encoding = self.compress_depth_enc(dpt_enc_last.float()).unsqueeze(1)  # [B, 1, goal_encoding_size]
            
            # Concatenate: obs + depth + goal
            all_encodings = torch.cat([obs_encoding, depth_encoding, goal_encoding], dim=1)
        else:
            # Concatenate: obs + goal
            all_encodings = torch.cat([obs_encoding, goal_encoding], dim=1)

        # ============ Apply Attention Mechanism ============
        # Set up goal masking if needed
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(
                self.all_masks.to(device), 0, no_goal_mask
            )
        else:
            src_key_padding_mask = None

        # Apply positional encoding
        if self.positional_encoding:
            all_encodings = self.positional_encoding(all_encodings)

        # Apply self-attention
        encoding_tokens = self.sa_encoder(
            all_encodings, src_key_padding_mask=src_key_padding_mask
        )
        
        # Apply masking and average pooling
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(
                self.avg_pool_mask.to(device), 0, no_goal_mask
            ).unsqueeze(-1)
            encoding_tokens = encoding_tokens * avg_mask

        # Apply DIFP: Dynamics-Inspired Feature Refinement
        # encoding_tokens: (B, num_tokens, obs_encoding_size)
        # Z_tilde: refined temporal features, z_c: global motion context
        Z_tilde, z_c = self.difp(encoding_tokens)  # (B, T, D), (B, D)

        # Global average pooling over refined tokens
        final_encoding = torch.mean(Z_tilde, dim=1)  # (B, obs_encoding_size)

        # Return both the encoding for conditioning and z_c for field estimation
        return final_encoding, z_c


# Utils for Group Norm (kept for potential future use)
def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, num_channels=x.num_features
        ),
    )
    return root_module


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module