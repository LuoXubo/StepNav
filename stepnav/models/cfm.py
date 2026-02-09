import torch
import torch.nn.functional as F
from conditional_flow_matching import ConditionalFlowMatcher


def compute_smoothness_loss(trajectory, action_mask=None):
    """Compute trajectory smoothness loss using jerk (third derivative) penalty.

    From the paper's Reg-CFM formulation:
        L_smooth = rho * ||dddot(tau_t)||^2

    Approximated via finite differences:
        jerk_k = x_{k+3} - 3*x_{k+2} + 3*x_{k+1} - x_k

    Args:
        trajectory: (B, T, A) trajectory tensor with T waypoints.
        action_mask: (B, T) optional mask of valid timesteps.

    Returns:
        Scalar smoothness loss.
    """
    batch_size, seq_len, action_dim = trajectory.shape
    if seq_len < 4:
        return torch.tensor(0.0, device=trajectory.device)

    smooth_loss = 0.0
    for k in range(seq_len - 3):
        jerk = (
            trajectory[:, k + 3]
            - 3 * trajectory[:, k + 2]
            + 3 * trajectory[:, k + 1]
            - trajectory[:, k]
        )
        loss_k = torch.sum(jerk ** 2, dim=-1)
        if action_mask is not None:
            mask_k = (
                action_mask[:, k]
                * action_mask[:, k + 1]
                * action_mask[:, k + 2]
                * action_mask[:, k + 3]
            )
            loss_k = loss_k * mask_k
        smooth_loss += loss_k.mean()
    return smooth_loss


def compute_safety_loss(trajectory, success_field_fn, epsilon=0.1, action_mask=None):
    """Compute safety loss using logarithmic barrier from the paper.

    From the Reg-CFM loss:
        L_safe = -kappa * sum_k log(max(d(x_k) - epsilon, 0))

    where d(x_k) is the distance to obstacles, approximated as 1 - F(x_k)
    when using the success probability field. We use F(x_k) directly as
    the distance proxy (higher F = safer).

    Args:
        trajectory: (B, T, A) trajectory tensor.
        success_field_fn: Callable mapping (B, A) waypoints -> (B,) distance proxy.
        epsilon: Safety margin threshold.
        action_mask: (B, T) optional validity mask.

    Returns:
        Scalar safety loss.
    """
    batch_size, seq_len, action_dim = trajectory.shape
    safe_loss = 0.0
    for k in range(seq_len):
        waypoint = trajectory[:, k]
        d_k = success_field_fn(waypoint)
        safe_distance = torch.clamp(d_k - epsilon, min=1e-8)
        loss_k = -torch.log(safe_distance)
        if action_mask is not None:
            loss_k = loss_k * action_mask[:, k]
        safe_loss += loss_k.mean()
    return safe_loss


def train_regularized_cfm(
    model,
    prior_trajectory,
    naction,
    obsgoal_cond,
    action_mask,
    success_field_fn=None,
    lambda_smooth=0.1,
    lambda_safe=0.01,
    epsilon=0.1,
):
    """Train with Regularized Conditional Flow Matching (Reg-CFM).

    From the paper, the Reg-CFM loss is:
        L_RegCFM = E_{tau,t}[ ||v_theta(tau_t, t, z_c) - u_t||^2
                              + rho * ||dddot(tau_t)||^2
                              - kappa * sum_k log(max(d(x_k) - epsilon, 0)) ]

    The key difference from standard CFM is:
    1. Initialization from structured prior tau_0 ~ p_prior (not Gaussian noise)
    2. Explicit smoothness regularization (jerk penalty)
    3. Explicit safety regularization (log-barrier on obstacle distance)

    This allows convergence in ~5 steps instead of >10 for noise-based methods.

    Args:
        model: Neural network with callable interface (task_name, sample, timestep, global_cond).
        prior_trajectory: (B, T, A) structured prior trajectory x0 from Field2Prior.
        naction: (B, T, A) target expert trajectory x1 (normalized).
        obsgoal_cond: (B, C) conditioning tensor from vision encoder.
        action_mask: (B, T) mask for valid timesteps.
        success_field_fn: Optional callable mapping waypoints -> distance proxy.
        lambda_smooth: Weight rho for smoothness regularization.
        lambda_safe: Weight kappa for safety regularization.
        epsilon: Safety margin for log-barrier.

    Returns:
        dict: Loss components {total_loss, flow_loss, smooth_loss, safe_loss}.
    """
    FM = ConditionalFlowMatcher(sigma=0.0)

    # Sample interpolated trajectory and conditional vector field
    # x0 = prior_trajectory (structured prior, NOT Gaussian noise)
    # x1 = naction (expert target)
    t, xt, ut = FM.sample_location_and_conditional_flow(
        x0=prior_trajectory, x1=naction
    )

    # Predict the velocity field v_theta
    vt = model(
        "noise_pred_net",
        sample=xt,
        timestep=t,
        global_cond=obsgoal_cond,
    )

    # Flow matching loss: ||v_theta - u_t||^2
    flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)

    # Reshape interpolated trajectory for regularization computation
    batch_size, seq_len, action_dim = naction.shape
    current_trajectory = xt.reshape(batch_size, seq_len, action_dim)

    # Smoothness regularization: rho * ||jerk||^2
    smooth_loss = compute_smoothness_loss(current_trajectory, action_mask)

    # Safety regularization: -kappa * sum log(max(d - epsilon, 0))
    if success_field_fn is not None:
        safe_loss = compute_safety_loss(
            current_trajectory,
            success_field_fn,
            epsilon=epsilon,
            action_mask=action_mask,
        )
    else:
        safe_loss = torch.tensor(0.0, device=naction.device)

    # Total Reg-CFM loss
    total_loss = flow_loss + lambda_smooth * smooth_loss + lambda_safe * safe_loss

    return {
        "total_loss": total_loss,
        "flow_loss": flow_loss,
        "smooth_loss": smooth_loss,
        "safe_loss": safe_loss,
    }


def action_reduce(loss, action_mask=None):
    """Reduce loss tensor considering an optional action mask.

    Args:
        loss: Element-wise loss tensor (..., action_dim) or (...,).
        action_mask: (B, T) optional validity mask.

    Returns:
        Scalar reduced loss.
    """
    if action_mask is None:
        return loss.mean()
    if len(loss.shape) > len(action_mask.shape):
        masked_loss = loss * action_mask.unsqueeze(-1)
    else:
        masked_loss = loss * action_mask
    return masked_loss.sum() / (action_mask.sum() + 1e-8)


def create_field_based_distance_fn(field_grid, grid_size=64, workspace_bounds=(-5, 5)):
    """Create a success-field-based distance function for safety regularization.

    From the paper: d(x_k) is approximated via 1 - F(x_k), but since F represents
    the success probability, we use F(x_k) directly as the safety proxy.

    Args:
        field_grid: (B, G, G) success probability field.
        grid_size: Resolution of the field grid.
        workspace_bounds: (min, max) workspace extent in meters.

    Returns:
        Callable: (B, 2) waypoint -> (B,) distance proxy.
    """
    extent = workspace_bounds[1] - workspace_bounds[0]

    def distance_fn(waypoint):
        """Map waypoint coordinates to success field values.

        Args:
            waypoint: (B, 2) waypoint in workspace coordinates.

        Returns:
            (B,) success probability at each waypoint location.
        """
        B = waypoint.shape[0]
        device = waypoint.device

        # Convert workspace coordinates to normalized grid coordinates [-1, 1]
        normalized = (waypoint - workspace_bounds[0]) / extent * 2.0 - 1.0

        # grid_sample expects (B, C, H, W) and grid (B, H_out, W_out, 2)
        field_input = field_grid.unsqueeze(1)  # (B, 1, G, G)
        grid_coords = normalized.view(B, 1, 1, 2)  # (B, 1, 1, 2)

        sampled = F.grid_sample(
            field_input, grid_coords, mode='bilinear',
            padding_mode='border', align_corners=True
        )
        return sampled.view(B)  # (B,)

    return distance_fn


if __name__ == "__main__":
    # Example usage demonstrating Reg-CFM with structured prior
    batch_size = 32
    seq_len = 8
    action_dim = 2

    # Structured prior (NOT Gaussian noise) - simulates Field2Prior output
    prior_trajectory = torch.randn(batch_size, seq_len, action_dim) * 0.5
    naction = torch.randn(batch_size, seq_len, action_dim)
    obsgoal_cond = torch.randn(batch_size, 256)
    action_mask = torch.ones(batch_size, seq_len)

    # Example field for safety loss
    field_grid = torch.rand(batch_size, 64, 64)
    distance_fn = create_field_based_distance_fn(field_grid)

    print("Reg-CFM module ready. Provide model for training.")
    print(f"  Prior trajectory shape: {prior_trajectory.shape}")
    print(f"  Target trajectory shape: {naction.shape}")
    print(f"  Condition shape: {obsgoal_cond.shape}")