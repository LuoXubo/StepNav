import torch
import torch.nn.functional as F
from conditional_flow_matching import ConditionalFlowMatcher


def compute_smoothness_loss(trajectory, action_mask=None):
    """
    Compute trajectory smoothness loss using a finite-difference approximation
    of jerk (third derivative).
    L_smooth = sum ||x_{k+3} - 3*x_{k+2} + 3*x_{k+1} - x_k||^2

    Args:
        trajectory (Tensor): (batch_size, seq_len, action_dim)
        action_mask (Tensor|None): (batch_size, seq_len) mask of valid steps.
    Returns:
        Tensor: scalar smoothness loss.
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
    """
    Compute safety loss using a logarithmic barrier:
    L_safe = -sum log(max(d_k - epsilon, 0))

    Args:
        trajectory (Tensor): (batch_size, seq_len, action_dim)
        success_field_fn (Callable): maps waypoint -> distance proxy tensor (batch_size,)
        epsilon (float): safety margin threshold.
        action_mask (Tensor|None): (batch_size, seq_len) validity mask.
    Returns:
        Tensor: scalar safety loss.
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
    noise,
    naction,
    obsgoal_cond,
    action_mask,
    success_field_fn,
    lambda_smooth=0.1,
    lambda_safe=0.01,
    epsilon=0.1,
):
    """
    Train with Conditional Flow Matching plus smoothness & safety regularization.

    Args:
        model: neural network with callable interface like model(task_name, sample, timestep, global_cond)
        noise (Tensor): initial noisy trajectory x0, shape (B, T, A)
        naction (Tensor): target expert trajectory x1, shape (B, T, A)
        obsgoal_cond (Tensor): conditioning tensor (B, C)
        action_mask (Tensor): (B, T) mask for valid steps
        success_field_fn (Callable): waypoint -> distance proxy tensor
        lambda_smooth (float): weight for smoothness regularization
        lambda_safe (float): weight for safety regularization
        epsilon (float): safety margin
    Returns:
        dict: loss components {total_loss, flow_loss, smooth_loss, safe_loss}
    """
    FM = ConditionalFlowMatcher(sigma=0.0)
    t, xt, ut = FM.sample_location_and_conditional_flow(x0=noise, x1=naction)

    vt = model(
        "noise_pred_net",
        sample=xt,
        timestep=t,
        global_cond=obsgoal_cond,
    )

    flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)

    # Infer trajectory shape directly from naction
    batch_size, seq_len, action_dim = naction.shape
    current_trajectory = xt.reshape(batch_size, seq_len, action_dim)

    smooth_loss = compute_smoothness_loss(current_trajectory, action_mask)
    safe_loss = compute_safety_loss(
        current_trajectory,
        success_field_fn,
        epsilon=epsilon,
        action_mask=action_mask,
    )

    total_loss = flow_loss + lambda_smooth * smooth_loss + lambda_safe * safe_loss
    return {
        "total_loss": total_loss,
        "flow_loss": flow_loss,
        "smooth_loss": smooth_loss,
        "safe_loss": safe_loss,
    }


def action_reduce(loss, action_mask=None):
    """
    Reduce loss considering an optional action mask.

    Args:
        loss (Tensor): element-wise loss (..., action_dim) or (...,)
        action_mask (Tensor|None): (batch_size, seq_len)
    Returns:
        Tensor: scalar reduced loss.
    """
    if action_mask is None:
        return loss.mean()
    if len(loss.shape) > len(action_mask.shape):
        masked_loss = loss * action_mask.unsqueeze(-1)
    else:
        masked_loss = loss * action_mask
    return masked_loss.sum() / (action_mask.sum() + 1e-8)


def example_success_field_fn(waypoint):
    """
    Example success field function.
    Returns a proxy distance to a fixed obstacle center.

    Args:
        waypoint (Tensor): (batch_size, action_dim)
    Returns:
        Tensor: (batch_size,) distances
    """
    obstacle_center = torch.tensor([5.0, 5.0], device=waypoint.device)
    distances = torch.norm(waypoint - obstacle_center, dim=-1)
    return distances


if __name__ == "__main__":
    # Example usage (model must be provided externally)
    batch_size = 32
    seq_len = 50
    action_dim = 2

    noise = torch.randn(batch_size, seq_len, action_dim)
    naction = torch.randn(batch_size, seq_len, action_dim)
    obsgoal_cond = torch.randn(batch_size, 256)
    action_mask = torch.ones(batch_size, seq_len)

    # Placeholder: model is required. This will raise if model is None.
    # Replace with an actual model implementing the expected interface.
    # loss_dict = train_regularized_cfm(
    #     model=your_model,
    #     noise=noise,
    #     naction=naction,
    #     obsgoal_cond=obsgoal_cond,
    #     action_mask=action_mask,
    #     success_field_fn=example_success_field_fn,
    #     lambda_smooth=0.1,
    #     lambda_safe=0.01,
    #     epsilon=0.1,
    # )
    # print(loss_dict)
    pass