import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import wandb
from diffusers.training_utils import EMAModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchvision import transforms
from stepnav.data.data_utils import VISUALIZATION_IMAGE_SIZE
from stepnav.models.cfm import (
    compute_smoothness_loss,
    compute_safety_loss,
    create_field_based_distance_fn,
)
from stepnav.training.logger import Logger
from stepnav.training.utils import (
    ACTION_STATS,
    action_reduce,
    compute_losses,
    get_delta,
    normalize_data,
    visualize_action_distribution,
    from_numpy,
)


def train(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    lambda_smooth: float = 0.1,
    lambda_safe: float = 0.01,
    lambda_field: float = 0.1,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """Train the StepNav model for one epoch.

    The training pipeline follows the paper's three-stage approach:
    1. Vision encoder with DIFP produces refined features and z_c.
    2. Field2Prior predicts success field F and extracts structured prior tau_0.
    3. Reg-CFM refines tau_0 toward expert trajectory with smoothness + safety loss.

    Args:
        model: The full NoMaD model with vision encoder, Field2Prior, and U-Net.
        ema_model: Exponential moving average of the model.
        optimizer: Optimizer instance.
        dataloader: Training data loader.
        transform: Image transformation pipeline.
        device: Torch device.
        goal_mask_prob: Probability of masking the goal during training.
        project_folder: Path to save logs and checkpoints.
        epoch: Current epoch number.
        alpha: Weight for distance prediction loss.
        lambda_smooth: Weight for Reg-CFM smoothness regularization.
        lambda_safe: Weight for Reg-CFM safety regularization.
        lambda_field: Weight for success field supervision loss.
        print_log_freq: Frequency for printing logs.
        wandb_log_freq: Frequency for wandb logging.
        image_log_freq: Frequency for image logging.
        num_images_log: Number of images to log.
        use_wandb: Whether to use Weights & Biases logging.
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger(
        "uc_action_loss", "train", window_size=print_log_freq
    )
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger(
        "gc_action_loss", "train", window_size=print_log_freq
    )
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(
        dataloader,
        desc=f"Train epoch {epoch}",
        leave=True,
        dynamic_ncols=True,
        colour="magenta",
    ) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                goal_image,
                actions,
                distance,
                goal_pos,
                _,
                action_mask,
            ) = data

            # Split the observation image into RGB channels
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(
                obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]
            )
            batch_viz_goal_images = TF.resize(
                goal_image, VISUALIZATION_IMAGE_SIZE[::-1]
            )
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)

            # Get action mask
            action_mask = action_mask.to(device)

            # Get naction and normalize it
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)

            # Get batch size
            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond, traj_prior = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            # Get distance label
            distance = distance.float().to(device)

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (
                1e-2 + (1 - goal_mask.float()).mean()
            )

            # ====== Reg-CFM: Flow matching with structured prior ======
            # Use structured prior from Field2Prior as x0 (NOT Gaussian noise)
            # Resize prior trajectory to match action dimensions if needed
            if traj_prior.shape[1] != naction.shape[1]:
                # Interpolate prior to match action sequence length
                traj_prior_interp = F.interpolate(
                    traj_prior.permute(0, 2, 1),
                    size=naction.shape[1],
                    mode='linear',
                    align_corners=True,
                ).permute(0, 2, 1)
            else:
                traj_prior_interp = traj_prior

            # Conditional Flow Matching: interpolate between structured prior and expert
            FM = ConditionalFlowMatcher(sigma=0.0)
            t, xt, ut = FM.sample_location_and_conditional_flow(
                x0=traj_prior_interp, x1=naction
            )

            # Predict velocity field v_theta
            vt = model(
                "noise_pred_net", sample=xt, timestep=t, global_cond=obsgoal_cond
            )

            # Flow matching loss: ||v_theta - u_t||^2
            flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)

            # Smoothness regularization: rho * ||jerk(tau_t)||^2
            current_traj = xt.reshape(B, naction.shape[1], naction.shape[2])
            smooth_loss = compute_smoothness_loss(current_traj, action_mask)

            # Safety regularization using success field (if available)
            safe_loss = torch.tensor(0.0, device=device)
            if hasattr(model, '_last_field_grid') and model._last_field_grid is not None:
                field_distance_fn = create_field_based_distance_fn(
                    model._last_field_grid
                )
                safe_loss = compute_safety_loss(
                    current_traj, field_distance_fn, epsilon=0.1, action_mask=action_mask
                )

            # Success field supervision loss (biharmonic-regularized)
            field_loss = torch.tensor(0.0, device=device)
            if hasattr(model, '_last_field_grid') and model._last_field_grid is not None:
                field_grid = model._last_field_grid
                # Use expert actions as supervision for the field
                expert_grid_paths = actions[:, :, :2].to(device)
                if hasattr(model, 'field2prior') and model.field2prior is not None:
                    field_loss = model.field2prior.compute_field_loss(
                        field_grid, expert_grid_paths
                    )

            # Total loss: Reg-CFM + distance + field supervision
            loss = (
                alpha * dist_loss
                + (1 - alpha) * flow_loss
                + lambda_smooth * smooth_loss
                + lambda_safe * safe_loss
                + lambda_field * field_loss
            )

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            if use_wandb:
                wandb.log({"total_loss": loss_cpu})
                wandb.log({"dist_loss": dist_loss.item()})
                wandb.log({"flow_loss": flow_loss.item()})
                wandb.log({"smooth_loss": smooth_loss.item()})
                wandb.log({"safe_loss": safe_loss.item()})
                wandb.log({"field_loss": field_loss.item()})

            if i % print_log_freq == 0:
                losses = compute_losses(
                    ema_model=ema_model.averaged_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_dist_label=distance.to(device),
                    batch_action_label=actions.to(device),
                    device=device,
                    action_mask=action_mask.to(device),
                    use_wandb=use_wandb,
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(
                            f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                        )

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_action_distribution(
                    ema_model=ema_model.averaged_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_viz_obs_images=batch_viz_obs_images,
                    batch_viz_goal_images=batch_viz_goal_images,
                    batch_action_label=actions,
                    batch_distance_labels=distance,
                    batch_goal_pos=goal_pos,
                    device=device,
                    eval_type="train",
                    project_folder=project_folder,
                    epoch=epoch,
                    num_images_log=num_images_log,
                    num_samples=4,
                    use_wandb=use_wandb,
                )
