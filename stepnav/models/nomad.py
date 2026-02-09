import torch.nn as nn


class NoMaD(nn.Module):
    """NoMaD navigation model with structured trajectory prior.

    This is the top-level model integrating:
    1. Vision encoder (ViNT or V-JEPA based) with DIFP refinement
    2. Field2Prior module for success field prediction and prior extraction
    3. Noise prediction network (conditional U-Net) for Reg-CFM refinement
    4. Distance prediction network for goal distance estimation

    The forward method dispatches to the appropriate submodule based on func_name.

    Args:
        vision_encoder: Encoder module (NoMaD_ViNT or NoMaD_VJEPA with DIFP).
        noise_pred_net: Conditional U-Net for velocity field prediction in Reg-CFM.
        dist_pred_net: Dense network for distance-to-goal prediction.
        field2prior: Field2Prior module for structured prior generation.
    """

    def __init__(self, vision_encoder, noise_pred_net, dist_pred_net, field2prior=None):
        super(NoMaD, self).__init__()

        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
        self.field2prior = field2prior

    def forward(self, func_name, **kwargs):
        """Dispatch forward call to the appropriate submodule.

        Supported func_name values:
            - "vision_encoder": Encode observations and goal, produce conditioning
              and structured trajectory prior via DIFP + Field2Prior.
            - "noise_pred_net": Predict velocity field for Reg-CFM refinement.
            - "dist_pred_net": Predict distance to goal.

        Args:
            func_name: String identifier for the target submodule.
            **kwargs: Arguments forwarded to the target submodule.

        Returns:
            Module-specific output (see individual submodule docs).
        """
        if func_name == "vision_encoder":
            # Vision encoder now returns (encoding, z_c) thanks to DIFP
            encoder_output = self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs["input_goal_mask"],
            )

            # Handle both old (single tensor) and new (tuple) encoder outputs
            if isinstance(encoder_output, tuple):
                obsgoal_cond, z_c = encoder_output
            else:
                obsgoal_cond = encoder_output
                z_c = None

            # Generate structured prior via Field2Prior if available
            if self.field2prior is not None and z_c is not None:
                prior_traj, field_grid, all_trajs, mixture_weights = (
                    self.field2prior(obsgoal_cond, z_c)
                )
                # Store field and prior info for loss computation
                self._last_field_grid = field_grid
                self._last_all_trajs = all_trajs
                self._last_mixture_weights = mixture_weights
                output = (obsgoal_cond, prior_traj)
            else:
                output = obsgoal_cond

        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=kwargs["global_cond"],
            )
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError(f"Unknown func_name: {func_name}")
        return output


class DenseNetwork(nn.Module):
    """Dense network for scalar prediction (e.g., distance to goal).

    A simple MLP that maps a high-dimensional embedding to a scalar output.

    Args:
        embedding_dim: Dimension of the input embedding vector.
    """

    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 16, 1),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, embedding_dim) or (B, ...) input tensor.

        Returns:
            (B, 1) scalar predictions.
        """
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output
