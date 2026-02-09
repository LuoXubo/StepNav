import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
import heapq
from typing import List, Tuple, Optional


class SuccessFieldHead(nn.Module):
    """Convolutional head for predicting the success probability field F(x).

    Takes refined temporal features (Z_tilde) and the global motion context (z_c)
    to predict a dense success probability field on a 2D grid. The field is
    learned as the minimizer of a biharmonic-regularized variational energy:

        E[F] = integral( (F-y)^2 + mu*||grad F||^2 + nu*||grad^2 F||^2 ) dx

    The biharmonic term penalizes sharp transitions, producing smooth navigable
    corridors from sparse expert demonstrations.

    Args:
        feature_dim: Dimension of the input feature vectors.
        context_dim: Dimension of the global motion context z_c.
        grid_size: Resolution of the output field grid (grid_size x grid_size).
        hidden_dim: Hidden layer dimension for the prediction network.
    """

    def __init__(self, feature_dim=256, context_dim=256, grid_size=64, hidden_dim=256):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim

        # Fuse temporal features and motion context into a single representation
        self.feature_fuse = nn.Sequential(
            nn.Linear(feature_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Decode fused features into a spatial feature map
        self.field_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4 * 4),
            nn.ReLU(),
        )

        # Transposed convolution decoder: (hidden_dim, 4, 4) -> (1, 64, 64)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z_pooled, z_c):
        """Predict success probability field.

        Args:
            z_pooled: (B, feature_dim) pooled temporal features.
            z_c: (B, context_dim) global motion context from DIFP.

        Returns:
            field: (B, grid_size, grid_size) success probability values in [0, 1].
        """
        B = z_pooled.shape[0]

        # Fuse features with motion context
        fused = self.feature_fuse(torch.cat([z_pooled, z_c], dim=-1))

        # Decode to spatial feature map
        x = self.field_decoder(fused)
        x = x.view(B, self.hidden_dim, 4, 4)
        x = self.conv_decoder(x)  # (B, 1, H, W)

        # Resize to target grid if needed
        if x.shape[-1] != self.grid_size:
            x = F.interpolate(
                x, size=(self.grid_size, self.grid_size),
                mode='bilinear', align_corners=False
            )

        # Apply sigmoid to get probabilities in [0, 1]
        field = torch.sigmoid(x.squeeze(1))  # (B, grid_size, grid_size)
        return field


class Field2Prior(nn.Module):
    """Convert success probability field to structured multi-modal trajectory prior.

    This module implements the core prior generation pipeline from the paper:
    1. Predict a dense success probability field F from refined features.
    2. Construct an energy landscape E(tau) = integral(-log(F(tau(t))+delta) ||tau'(t)||) dt.
    3. Extract K diverse candidate paths via shortest-path search on the energy.
    4. Select a diverse subset using greedy max-min Hausdorff distance criterion.
    5. Assign mixture weights: pi_m proportional to exp(S(tau^(m)) / temperature).

    Args:
        feature_dim: Dimension of the input feature vector (from vision encoder).
        context_dim: Dimension of the global motion context z_c.
        grid_size: Resolution of the success field grid.
        workspace_bounds: (min, max) workspace extent in meters.
        num_waypoints: Maximum number of salient peaks to detect via NMS.
        num_paths: Number of candidate paths to generate.
        num_final_trajectories: Number of diverse trajectories in the mixture prior.
        trajectory_length: Number of waypoints per trajectory.
        temperature: Softmax temperature for mixture weight computation.
        delta: Small constant to avoid log(0) in energy computation.
    """

    def __init__(
        self,
        feature_dim=256,
        context_dim=256,
        grid_size=64,
        workspace_bounds=(-5, 5),
        num_waypoints=5,
        num_paths=10,
        num_final_trajectories=3,
        trajectory_length=20,
        temperature=0.1,
        delta=1e-6,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.grid_size = grid_size
        self.workspace_bounds = workspace_bounds
        self.num_waypoints = num_waypoints
        self.num_paths = num_paths
        self.num_final_trajectories = num_final_trajectories
        self.trajectory_length = trajectory_length
        self.temperature = temperature
        self.delta = delta

        # Success field prediction head
        self.field_head = SuccessFieldHead(
            feature_dim=feature_dim,
            context_dim=context_dim,
            grid_size=grid_size,
        )

    def forward(self, z_pooled, z_c):
        """Generate structured multi-modal trajectory prior.

        Args:
            z_pooled: (B, feature_dim) pooled visual features from encoder.
            z_c: (B, context_dim) global motion context from DIFP module.

        Returns:
            prior_trajectory: (B, trajectory_length, 2) sampled prior trajectory.
            field_grid: (B, grid_size, grid_size) predicted success field.
            all_trajectories: (B, M, trajectory_length, 2) all candidate trajectories.
            mixture_weights: (B, M) mixture weights for each candidate.
        """
        B = z_pooled.shape[0]

        # Step 1: Predict success probability field
        field_grid = self.field_head(z_pooled, z_c)  # (B, G, G)

        # Step 2-5: Extract structured prior for each batch element
        all_trajs_list = []
        all_weights_list = []
        sampled_prior_list = []

        for b in range(B):
            field = field_grid[b]  # (G, G)

            # Find salient peaks via non-maximum suppression
            waypoints = self._find_waypoints(field)

            # Generate diverse candidate paths through low-energy corridors
            paths = self._generate_paths(field, waypoints)

            # Smooth discrete paths into continuous trajectories via B-splines
            smooth_trajectories = self._smooth_paths(paths)

            # Select diverse subset using max-min Hausdorff criterion
            final_trajs, traj_weights = self._select_diverse_trajectories(
                smooth_trajectories, field
            )

            all_trajs_list.append(final_trajs)
            all_weights_list.append(traj_weights)

            # Sample one prior trajectory according to mixture weights
            sampled_idx = torch.multinomial(traj_weights, 1).item()
            sampled_prior_list.append(final_trajs[sampled_idx])

        # Stack results across batch
        all_trajectories = torch.stack(all_trajs_list)   # (B, M, L, 2)
        mixture_weights = torch.stack(all_weights_list)   # (B, M)
        prior_trajectory = torch.stack(sampled_prior_list)  # (B, L, 2)

        return prior_trajectory, field_grid, all_trajectories, mixture_weights

    def _find_waypoints(self, field):
        """Find salient peaks in the success field via non-maximum suppression.

        Args:
            field: (G, G) success probability field tensor.

        Returns:
            waypoints: (K, 2) array of peak grid coordinates.
        """
        field_np = field.detach().cpu().numpy()
        from scipy.ndimage import gaussian_filter, maximum_filter

        smoothed = gaussian_filter(field_np, sigma=1.0)

        # Non-maximum suppression: keep only local maxima above threshold
        local_max = maximum_filter(smoothed, size=5)
        peaks = (smoothed == local_max) & (smoothed > 0.3)

        peak_coords = np.argwhere(peaks)
        peak_values = smoothed[peaks]

        if len(peak_coords) > 0:
            indices = np.argsort(peak_values)[-self.num_waypoints:]
            waypoints = peak_coords[indices]
        else:
            # Fallback: uniformly distributed points around center
            center = self.grid_size // 2
            waypoints = np.array([
                [center + int(10 * np.cos(2 * np.pi * i / self.num_waypoints)),
                 center + int(10 * np.sin(2 * np.pi * i / self.num_waypoints))]
                for i in range(self.num_waypoints)
            ])
            waypoints = np.clip(waypoints, 0, self.grid_size - 1)

        return waypoints

    def _generate_paths(self, field, waypoints):
        """Generate diverse candidate paths through low-energy corridors.

        Energy landscape: E(tau) = -log(F(tau) + delta), so minimizing E
        maximizes cumulative success probability.

        Args:
            field: (G, G) success probability field tensor.
            waypoints: (K, 2) array of target waypoint grid coordinates.

        Returns:
            paths: List of discrete paths (lists of (y, x) grid tuples).
        """
        field_np = field.detach().cpu().numpy()
        energy = -np.log(field_np + self.delta)

        paths = []
        start = np.array([self.grid_size // 2, self.grid_size // 2])

        for wp in waypoints[:self.num_paths]:
            # Generate multiple variants per waypoint for diversity
            for noise_std in [0.0, 0.1, 0.2]:
                path = self._astar_search(energy, start, wp, noise_std=noise_std)
                if path is not None and len(path) > 2:
                    paths.append(path)

        return paths

    def _astar_search(self, energy, start, goal, noise_std=0.0):
        """A* pathfinding on the energy landscape with optional noise.

        Args:
            energy: (H, W) energy grid.
            start: (2,) start grid coordinates.
            goal: (2,) goal grid coordinates.
            noise_std: Noise std for path diversity (approximates K-shortest paths).

        Returns:
            path: List of (y, x) tuples, or None if no path found.
        """
        H, W = energy.shape

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        def neighbors(pos):
            y, x = pos
            candidates = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        candidates.append((ny, nx))
            return candidates

        start = tuple(start)
        goal = tuple(goal)
        
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                # Reconstruct path from goal to start
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))
            
            for next_pos in neighbors(current):
                # Edge cost: energy * step distance + optional noise
                noise = np.random.normal(0, noise_std) if noise_std > 0 else 0
                step_dist = np.sqrt(
                    (next_pos[0] - current[0]) ** 2 + (next_pos[1] - current[1]) ** 2
                )
                new_cost = cost_so_far[current] + energy[next_pos] * step_dist + noise
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None
    
    def _smooth_paths(self, paths):
        """Convert discrete grid paths to smooth trajectories via B-spline fitting.

        Args:
            paths: List of discrete paths (lists of (y, x) grid tuples).

        Returns:
            smooth_trajectories: List of (trajectory_length, 2) numpy arrays
                in workspace coordinates.
        """
        smooth_trajectories = []
        
        for path in paths:
            if len(path) < 4:  # Need at least 4 points for cubic spline
                continue
                
            # Convert grid indices to workspace coordinates
            path_coords = []
            extent = self.workspace_bounds[1] - self.workspace_bounds[0]
            for y, x in path:
                wx = self.workspace_bounds[0] + (x / self.grid_size) * extent
                wy = self.workspace_bounds[0] + (y / self.grid_size) * extent
                path_coords.append([wx, wy])
            
            path_coords = np.array(path_coords)
            
            try:
                # Fit cubic B-spline with moderate smoothing
                tck, u = splprep([path_coords[:, 0], path_coords[:, 1]], s=0.5, k=3)
                
                # Resample at uniform parameter intervals
                u_new = np.linspace(0, 1, self.trajectory_length)
                x_new, y_new = splev(u_new, tck)
                
                trajectory = np.stack([x_new, y_new], axis=-1)
                smooth_trajectories.append(trajectory)
            except Exception:
                # Fallback: linear interpolation if spline fitting fails
                indices = np.linspace(0, len(path_coords)-1, self.trajectory_length)
                trajectory = np.array([
                    np.interp(indices, range(len(path_coords)), path_coords[:, i])
                    for i in range(2)
                ]).T
                smooth_trajectories.append(trajectory)
        
        return smooth_trajectories

    def _compute_trajectory_score(self, traj, field):
        """Compute the score S(tau) for a trajectory based on the energy landscape.

        S(tau) = -E(tau) = integral( log(F(tau(t)) + delta) * ||tau'(t)|| ) dt

        Args:
            traj: (L, 2) numpy array of trajectory waypoints in workspace coords.
            field: (G, G) success field tensor.

        Returns:
            score: Scalar trajectory score (higher is better).
        """
        traj_grid = self._coords_to_grid_indices(traj)
        valid_mask = (
            (traj_grid[:, 0] >= 0) & (traj_grid[:, 0] < self.grid_size) &
            (traj_grid[:, 1] >= 0) & (traj_grid[:, 1] < self.grid_size)
        )

        if valid_mask.sum() > 0:
            field_values = field[traj_grid[valid_mask, 0], traj_grid[valid_mask, 1]]
            avg_log_prob = torch.log(field_values + self.delta).mean().item()
        else:
            avg_log_prob = -10.0

        # Path length penalty
        length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

        # Curvature penalty
        if len(traj) > 2:
            vel = np.diff(traj, axis=0)
            acc = np.diff(vel, axis=0)
            curvature = np.mean(np.linalg.norm(acc, axis=1))
        else:
            curvature = 0.0

        score = avg_log_prob - 0.1 * length - 0.05 * curvature
        return score
    
    def _select_diverse_trajectories(self, trajectories, field):
        """Select diverse trajectory subset using greedy max-min Hausdorff distance.

        From the paper: "We select a diverse subset T_prior using a greedy
        max-min Hausdorff criterion" to preserve multi-modality.

        Mixture weights: pi_m proportional to exp(S(tau^(m)) / temperature)

        Args:
            trajectories: List of (L, 2) numpy trajectory arrays.
            field: (G, G) success field tensor.

        Returns:
            final_trajs: (M, L, 2) tensor of selected trajectories.
            weights: (M,) tensor of mixture weights.
        """
        device = field.device

        if len(trajectories) == 0:
            # Fallback: generate straight lines in diverse directions
            trajs = []
            for i in range(self.num_final_trajectories):
                angle = 2 * np.pi * i / self.num_final_trajectories
                end_point = np.array([np.cos(angle), np.sin(angle)]) * 3.0
                traj = np.linspace([0, 0], end_point, self.trajectory_length)
                trajs.append(traj)
            
            trajs_tensor = torch.tensor(trajs, dtype=torch.float32, device=device)
            weights = torch.ones(self.num_final_trajectories, device=device)
            weights = F.softmax(weights / self.temperature, dim=0)
            return trajs_tensor, weights
        
        # Compute scores for all candidate trajectories
        scores = [self._compute_trajectory_score(t, field) for t in trajectories]
        
        # Greedy max-min Hausdorff selection
        selected = []
        remaining = list(range(len(trajectories)))
        
        # First: select the highest-scoring trajectory
        best_idx = remaining[np.argmax([scores[i] for i in remaining])]
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Subsequent: maximize minimum Hausdorff distance to selected set
        while len(selected) < min(self.num_final_trajectories, len(trajectories)):
            max_min_dist = -1
            best_idx = None
            
            for idx in remaining:
                min_dist = float('inf')
                for sel_idx in selected:
                    dist = directed_hausdorff(trajectories[idx], trajectories[sel_idx])[0]
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        # Pad with last selected if fewer than num_final_trajectories
        while len(selected) < self.num_final_trajectories:
            selected.append(selected[-1])
        
        # Convert to tensors
        final_trajs = torch.tensor(
            np.array([trajectories[i] for i in selected[:self.num_final_trajectories]]),
            dtype=torch.float32,
            device=field.device
        )
        
        # Compute mixture weights: pi_m = softmax(S(tau^(m)) / temperature)
        selected_scores = torch.tensor(
            [scores[i] for i in selected[:self.num_final_trajectories]],
            dtype=torch.float32,
            device=field.device
        )
        weights = F.softmax(selected_scores / self.temperature, dim=0)
        
        return final_trajs, weights
    
    def _coords_to_grid_indices(self, coords):
        """Convert workspace coordinates to grid indices.

        Args:
            coords: (N, 2) numpy array of (x, y) workspace coordinates.

        Returns:
            indices: (N, 2) numpy array of (row, col) grid indices.
        """
        indices = []
        extent = self.workspace_bounds[1] - self.workspace_bounds[0]
        for x, y in coords:
            col = int((x - self.workspace_bounds[0]) / extent * self.grid_size)
            row = int((y - self.workspace_bounds[0]) / extent * self.grid_size)
            indices.append([row, col])
        return np.array(indices)
    
    def compute_field_loss(self, field_pred, expert_paths, mu=0.01, nu=0.001):
        """Compute training loss for the success probability field.

        The loss is derived from the variational energy in the paper:
            E[F] = (F - y)^2 + mu * ||grad F||^2 + nu * ||grad^2 F||^2

        The Euler-Lagrange equation yields a biharmonic-regularized Poisson PDE:
            -nu * Delta^2 F + mu * Delta F + (F - y) = 0

        We approximate the regularization terms using finite differences.

        Args:
            field_pred: (B, G, G) predicted success probabilities.
            expert_paths: (B, L, 2) expert trajectory waypoints in grid coordinates.
            mu: Weight for gradient (Laplacian) smoothness regularization.
            nu: Weight for biharmonic regularization.

        Returns:
            total_loss: Scalar training loss.
        """
        B, G, _ = field_pred.shape

        # Create binary labels from expert demonstrations
        labels = torch.zeros_like(field_pred)
        for b in range(B):
            if b < len(expert_paths) and expert_paths[b] is not None:
                for point in expert_paths[b]:
                    row = int(point[1].item() if isinstance(point[1], torch.Tensor)
                              else point[1])
                    col = int(point[0].item() if isinstance(point[0], torch.Tensor)
                              else point[0])
                    row = max(0, min(row, G - 1))
                    col = max(0, min(col, G - 1))
                    # Gaussian label spreading
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            r, c = row + dr, col + dc
                            if 0 <= r < G and 0 <= c < G:
                                dist_sq = dr ** 2 + dc ** 2
                                labels[b, r, c] = max(
                                    labels[b, r, c].item(),
                                    np.exp(-dist_sq / 2.0)
                                )

        # Data fidelity: BCE loss
        bce_loss = F.binary_cross_entropy(field_pred, labels)

        # Laplacian regularization: mu * ||grad F||^2
        laplacian = torch.zeros_like(field_pred)
        laplacian[:, 1:-1, 1:-1] = (
            field_pred[:, :-2, 1:-1] + field_pred[:, 2:, 1:-1] +
            field_pred[:, 1:-1, :-2] + field_pred[:, 1:-1, 2:] -
            4 * field_pred[:, 1:-1, 1:-1]
        )
        grad_loss = (laplacian ** 2).mean()

        # Biharmonic regularization: nu * ||grad^2 F||^2
        # Approximated as Laplacian of Laplacian
        biharmonic = torch.zeros_like(field_pred)
        biharmonic[:, 2:-2, 2:-2] = (
            laplacian[:, 1:-3, 2:-2] + laplacian[:, 3:-1, 2:-2] +
            laplacian[:, 2:-2, 1:-3] + laplacian[:, 2:-2, 3:-1] -
            4 * laplacian[:, 2:-2, 2:-2]
        )
        biharmonic_loss = (biharmonic ** 2).mean()

        total_loss = bce_loss + mu * grad_loss + nu * biharmonic_loss
        return total_loss


# Example usage and testing
if __name__ == "__main__":
    model = Field2Prior(
        feature_dim=256,
        context_dim=256,
        num_final_trajectories=3,
        trajectory_length=20
    )
    
    batch_size = 4
    z_pooled = torch.randn(batch_size, 256)
    z_c = torch.randn(batch_size, 256)
    
    prior_traj, field, all_trajs, weights = model(z_pooled, z_c)
    
    print(f"Prior trajectory shape: {prior_traj.shape}")     # (4, 20, 2)
    print(f"Field shape: {field.shape}")                     # (4, 64, 64)
    print(f"All trajectories shape: {all_trajs.shape}")      # (4, 3, 20, 2)
    print(f"Mixture weights shape: {weights.shape}")         # (4, 3)
    print(f"Weights sum per batch: {weights.sum(dim=1)}")    # ~1.0