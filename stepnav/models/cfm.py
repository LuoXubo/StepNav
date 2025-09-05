import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

class RegularizedConditionalFlowMatching:
    """
    Regularized Conditional Flow Matching for trajectory refinement
    with smoothness and safety regularizers.
    """
    
    def __init__(
        self, 
        sigma: float = 0.0,
        smooth_weight: float = 1.0,
        safety_weight: float = 1.0,
        safety_epsilon: float = 0.1
    ):
        """
        Args:
            sigma: Noise level for flow matching
            smooth_weight: Weight for smoothness regularizer
            safety_weight: Weight for safety regularizer
            safety_epsilon: Safety margin for obstacle avoidance
        """
        self.sigma = sigma
        self.smooth_weight = smooth_weight
        self.safety_weight = safety_weight
        self.safety_epsilon = safety_epsilon
        
    def sample_location_and_conditional_flow(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample time t and corresponding trajectory point xt with conditional flow ut.
        
        Args:
            x0: Initial trajectory (batch_size, seq_len, dim)
            x1: Target expert trajectory (batch_size, seq_len, dim)
            
        Returns:
            t: Sampled time (batch_size, 1)
            xt: Interpolated trajectory at time t (batch_size, seq_len, dim)
            ut: Conditional flow at time t (batch_size, seq_len, dim)
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample time uniformly from [0, 1]
        t = torch.rand(batch_size, 1, device=device)
        
        # Linear interpolation: xt = (1-t)*x0 + t*x1 + sigma*noise
        t_expanded = t.unsqueeze(-1)  # (batch_size, 1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        if self.sigma > 0:
            noise = torch.randn_like(xt) * self.sigma
            xt = xt + noise
        
        # Conditional flow: ut = x1 - x0 (+ noise derivative if sigma > 0)
        ut = x1 - x0
        
        return t.squeeze(-1), xt, ut
    
    def compute_smoothness_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularizer based on finite-difference jerk approximation.
        
        Args:
            trajectory: Trajectory tensor (batch_size, seq_len, dim)
            
        Returns:
            Smoothness loss scalar
        """
        if trajectory.shape[1] < 4:  # Need at least 4 points for jerk calculation
            return torch.tensor(0.0, device=trajectory.device)
        
        # Finite difference approximation: x_{k+3} - 3*x_{k+2} + 3*x_{k+1} - x_k
        x_k = trajectory[:, :-3, :]      # (batch_size, seq_len-3, dim)
        x_k1 = trajectory[:, 1:-2, :]    # (batch_size, seq_len-3, dim)
        x_k2 = trajectory[:, 2:-1, :]    # (batch_size, seq_len-3, dim)
        x_k3 = trajectory[:, 3:, :]      # (batch_size, seq_len-3, dim)
        
        jerk_approx = x_k3 - 3 * x_k2 + 3 * x_k1 - x_k
        smooth_loss = torch.sum(jerk_approx ** 2)
        
        return smooth_loss
    
    def compute_safety_loss(
        self, 
        trajectory: torch.Tensor, 
        success_field_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute safety log-barrier regularizer based on distance to obstacles.
        
        Args:
            trajectory: Trajectory tensor (batch_size, seq_len, dim)
            success_field_fn: Function that takes waypoints and returns success field values
            
        Returns:
            Safety loss scalar
        """
        batch_size, seq_len, dim = trajectory.shape
        
        # Flatten trajectory to evaluate success field
        waypoints = trajectory.view(-1, dim)  # (batch_size * seq_len, dim)
        
        # Get success field values (proxy for distance to obstacles)
        success_values = success_field_fn(waypoints)  # (batch_size * seq_len,)
        
        # Reshape back
        success_values = success_values.view(batch_size, seq_len)
        
        # Compute distance estimates (proportional to success field)
        distances = success_values  # d_k ∝ F(x_k)
        
        # Log-barrier: -log(max(d_k - epsilon, small_value))
        barrier_terms = torch.clamp(distances - self.safety_epsilon, min=1e-8)
        safety_loss = -torch.sum(torch.log(barrier_terms))
        
        return safety_loss
    
    def compute_total_loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor,
        success_field_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss including flow matching and regularizers.
        
        Args:
            model: Vector field model v_θ(τ_t, t, z_c)
            x0: Initial trajectory
            x1: Target expert trajectory  
            condition: Conditioning information z_c
            success_field_fn: Function to compute success field values
            action_mask: Mask for valid actions
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Sample flow matching data
        t, xt, ut = self.sample_location_and_conditional_flow(x0, x1)
        
        # Predict vector field
        vt = model(sample=xt, timestep=t, global_cond=condition)
        
        # Flow matching loss (L2 between predicted and true flow)
        if action_mask is not None:
            flow_loss = F.mse_loss(vt * action_mask.unsqueeze(-1), 
                                 ut * action_mask.unsqueeze(-1), 
                                 reduction='sum') / action_mask.sum()
        else:
            flow_loss = F.mse_loss(vt, ut, reduction='mean')
        
        # Initialize loss dictionary
        loss_dict = {'flow_loss': flow_loss}
        total_loss = flow_loss
        
        # Smoothness regularizer
        if self.smooth_weight > 0:
            smooth_loss = self.compute_smoothness_loss(xt)
            loss_dict['smooth_loss'] = smooth_loss
            total_loss = total_loss + self.smooth_weight * smooth_loss
        
        # Safety regularizer
        if self.safety_weight > 0 and success_field_fn is not None:
            safety_loss = self.compute_safety_loss(xt, success_field_fn)
            loss_dict['safety_loss'] = safety_loss
            total_loss = total_loss + self.safety_weight * safety_loss
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


class TrajectoryRefinementModel:
    """
    Complete trajectory refinement system using regularized conditional flow matching.
    """
    
    def __init__(
        self, 
        vector_field_model: torch.nn.Module,
        cfm_config: dict = None
    ):
        """
        Args:
            vector_field_model: Neural network model for vector field v_θ
            cfm_config: Configuration for ConditionalFlowMatcher
        """
        self.model = vector_field_model
        
        # Default CFM configuration
        default_config = {
            'sigma': 0.0,
            'smooth_weight': 1.0, 
            'safety_weight': 1.0,
            'safety_epsilon': 0.1
        }
        if cfm_config:
            default_config.update(cfm_config)
            
        self.cfm = RegularizedConditionalFlowMatching(**default_config)
    
    def train_step(
        self,
        initial_trajectory: torch.Tensor,
        expert_trajectory: torch.Tensor, 
        condition: torch.Tensor,
        success_field_fn: Optional[Callable] = None,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single training step for trajectory refinement.
        
        Args:
            initial_trajectory: τ_0 from structured mixture
            expert_trajectory: τ_1 expert trajectory
            condition: Conditioning information z_c
            success_field_fn: Function to evaluate success field
            action_mask: Mask for valid trajectory points
            
        Returns:
            loss: Total training loss
            metrics: Dictionary of loss components
        """
        return self.cfm.compute_total_loss(
            model=self.model,
            x0=initial_trajectory,
            x1=expert_trajectory,
            condition=condition,
            success_field_fn=success_field_fn,
            action_mask=action_mask
        )
    
    def refine_trajectory(
        self, 
        initial_trajectory: torch.Tensor,
        condition: torch.Tensor,
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Refine initial trajectory using learned vector field.
        
        Args:
            initial_trajectory: τ_0 to be refined
            condition: Conditioning information z_c
            num_steps: Number of ODE integration steps
            
        Returns:
            Refined trajectory
        """
        device = initial_trajectory.device
        dt = 1.0 / num_steps
        
        trajectory = initial_trajectory.clone()
        
        with torch.no_grad():
            for step in range(num_steps):
                t = torch.full((trajectory.shape[0],), step * dt, device=device)
                
                # Get vector field prediction
                velocity = self.model(sample=trajectory, timestep=t, global_cond=condition)
                
                # Euler integration: τ_{t+dt} = τ_t + dt * v_θ(τ_t, t, z_c)
                trajectory = trajectory + dt * velocity
        
        return trajectory


# Example usage
def example_usage():
    """
    Example of how to use the regularized conditional flow matching system.
    """
    import torch.nn as nn
    
    # Example vector field model
    class SimpleVectorField(nn.Module):
        def __init__(self, traj_dim, cond_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(traj_dim + 1 + cond_dim, hidden_dim),  # trajectory + time + condition
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, traj_dim)
            )
        
        def forward(self, sample, timestep, global_cond):
            # Flatten spatial dimensions for processing
            batch_size, seq_len, traj_dim = sample.shape
            sample_flat = sample.view(batch_size, -1)
            timestep_expanded = timestep.unsqueeze(-1).expand(batch_size, seq_len).contiguous().view(batch_size, -1)
            cond_expanded = global_cond.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size, -1)
            
            # Concatenate inputs
            input_tensor = torch.cat([sample_flat, timestep_expanded, cond_expanded], dim=-1)
            output = self.net(input_tensor)
            
            return output.view(batch_size, seq_len, traj_dim)
    
    # Example success field function
    def example_success_field(waypoints):
        """Simple example: distance from origin as proxy for obstacle distance"""
        return torch.norm(waypoints, dim=-1)
    
    # Setup
    batch_size, seq_len, traj_dim, cond_dim = 4, 10, 2, 8
    
    model = SimpleVectorField(traj_dim, cond_dim)
    refiner = TrajectoryRefinementModel(
        model, 
        cfm_config={
            'smooth_weight': 0.1,
            'safety_weight': 0.05,
            'safety_epsilon': 0.2
        }
    )
    
    # Generate example data
    initial_traj = torch.randn(batch_size, seq_len, traj_dim)
    expert_traj = torch.randn(batch_size, seq_len, traj_dim) 
    condition = torch.randn(batch_size, cond_dim)
    
    # Training step
    loss, metrics = refiner.train_step(
        initial_traj, expert_traj, condition, 
        success_field_fn=example_success_field
    )
    
    print(f"Training loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Trajectory refinement
    refined_traj = refiner.refine_trajectory(initial_traj, condition, num_steps=50)
    print(f"Refined trajectory shape: {refined_traj.shape}")

if __name__ == "__main__":
    example_usage()