import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
import heapq
from typing import List, Tuple, Optional

class SuccessFieldMLP(nn.Module):
    """Implicit neural network for success probability field F(x)"""
    def __init__(self, coord_dim=2, context_dim=256, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        in_dim = coord_dim + context_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, coords, context):
        """
        Args:
            coords: [B, N, 2] spatial coordinates
            context: [B, context_dim] context vector
        Returns:
            field: [B, N] success probability for each coordinate
        """
        B, N, _ = coords.shape
        context_expanded = context.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([coords, context_expanded], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logits)

class Field2Prior(nn.Module):
    """Convert success field to structured multi-modal trajectory prior"""
    
    def __init__(self, 
                 context_dim=256,
                 coord_dim=2,
                 grid_size=64,
                 workspace_bounds=(-5, 5),  # meters
                 num_waypoints=5,
                 num_paths=10,
                 num_final_trajectories=3,
                 trajectory_length=20,
                 temperature=0.1):
        super().__init__()
        
        self.context_dim = context_dim
        self.coord_dim = coord_dim
        self.grid_size = grid_size
        self.workspace_bounds = workspace_bounds
        self.num_waypoints = num_waypoints
        self.num_paths = num_paths
        self.num_final_trajectories = num_final_trajectories
        self.trajectory_length = trajectory_length
        self.temperature = temperature
        
        # Context processing from [B, T] to [B, context_dim]
        self.context_processor = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # Success field MLP
        self.field_mlp = SuccessFieldMLP(coord_dim, context_dim)
        
        # Create grid coordinates
        self.register_buffer('grid_coords', self._create_grid())
        
    def _create_grid(self):
        """Create egocentric grid coordinates"""
        x = torch.linspace(self.workspace_bounds[0], self.workspace_bounds[1], self.grid_size)
        y = torch.linspace(self.workspace_bounds[0], self.workspace_bounds[1], self.grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        return coords
    
    def forward(self, features):
        """
        Args:
            features: [B, T] spatiotemporal features
        Returns:
            trajectories: [B, num_final_trajectories, trajectory_length, 2]
            weights: [B, num_final_trajectories] mixture weights
        """
        B = features.shape[0]
        
        # Process context
        context = self.context_processor(features)  # [B, context_dim]
        
        # Compute success field on grid
        grid_coords = self.grid_coords.unsqueeze(0).expand(B, -1, -1)  # [B, G^2, 2]
        field_values = self.field_mlp(grid_coords, context)  # [B, G^2]
        field_grid = field_values.reshape(B, self.grid_size, self.grid_size)
        
        # Extract trajectories for each batch
        trajectories = []
        weights = []
        
        for b in range(B):
            field = field_grid[b]
            
            # Find waypoints via non-maximum suppression
            waypoints = self._find_waypoints(field)
            
            # Generate diverse paths
            paths = self._generate_paths(field, waypoints)
            
            # Convert paths to smooth trajectories
            smooth_trajectories = self._smooth_paths(paths)
            
            # Select diverse subset
            final_trajs, traj_weights = self._select_diverse_trajectories(
                smooth_trajectories, field
            )
            
            trajectories.append(final_trajs)
            weights.append(traj_weights)
        
        trajectories = torch.stack(trajectories)  # [B, M, L, 2]
        weights = torch.stack(weights)  # [B, M]
        
        return trajectories, weights
    
    def _find_waypoints(self, field):
        """Find salient peaks in success field via NMS"""
        # Apply gaussian smoothing
        field_np = field.detach().cpu().numpy()
        from scipy.ndimage import gaussian_filter, maximum_filter
        
        smoothed = gaussian_filter(field_np, sigma=1.0)
        
        # Non-maximum suppression
        local_max = maximum_filter(smoothed, size=5)
        peaks = (smoothed == local_max) & (smoothed > 0.3)
        
        # Get peak coordinates
        peak_coords = np.argwhere(peaks)
        peak_values = smoothed[peaks]
        
        # Sort by value and take top-k
        if len(peak_coords) > 0:
            indices = np.argsort(peak_values)[-self.num_waypoints:]
            waypoints = peak_coords[indices]
        else:
            # Fallback: sample random points
            waypoints = np.random.randint(0, self.grid_size, (self.num_waypoints, 2))
        
        return waypoints
    
    def _generate_paths(self, field, waypoints):
        """Generate diverse paths through low-energy corridors"""
        field_np = field.detach().cpu().numpy()
        energy = -np.log(field_np + 1e-6)
        
        paths = []
        
        # Add start point (robot position at origin in grid coords)
        start = np.array([self.grid_size // 2, self.grid_size // 2])
        
        for wp in waypoints[:self.num_paths]:
            # A* search with random perturbations for diversity
            for _ in range(2):  # Generate 2 variants per waypoint
                path = self._astar_search(energy, start, wp, noise_std=0.1)
                if path is not None and len(path) > 2:
                    paths.append(path)
        
        return paths
    
    def _astar_search(self, energy, start, goal, noise_std=0.0):
        """A* pathfinding with optional noise for diversity"""
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
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))
            
            for next_pos in neighbors(current):
                # Add noise to energy for diversity
                noise = np.random.normal(0, noise_std) if noise_std > 0 else 0
                new_cost = cost_so_far[current] + energy[next_pos] + noise
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None
    
    def _smooth_paths(self, paths):
        """Convert discrete paths to smooth trajectories using B-splines"""
        smooth_trajectories = []
        
        for path in paths:
            if len(path) < 4:  # Need at least 4 points for cubic spline
                continue
                
            # Convert grid indices to workspace coordinates
            path_coords = []
            for y, x in path:
                wx = self.workspace_bounds[0] + (x / self.grid_size) * \
                     (self.workspace_bounds[1] - self.workspace_bounds[0])
                wy = self.workspace_bounds[0] + (y / self.grid_size) * \
                     (self.workspace_bounds[1] - self.workspace_bounds[0])
                path_coords.append([wx, wy])
            
            path_coords = np.array(path_coords)
            
            try:
                # Fit B-spline
                tck, u = splprep([path_coords[:, 0], path_coords[:, 1]], s=0.5, k=3)
                
                # Sample points
                u_new = np.linspace(0, 1, self.trajectory_length)
                x_new, y_new = splev(u_new, tck)
                
                trajectory = np.stack([x_new, y_new], axis=-1)
                smooth_trajectories.append(trajectory)
            except:
                # Fallback: linear interpolation
                indices = np.linspace(0, len(path_coords)-1, self.trajectory_length)
                trajectory = np.array([
                    np.interp(indices, range(len(path_coords)), path_coords[:, i])
                    for i in range(2)
                ]).T
                smooth_trajectories.append(trajectory)
        
        return smooth_trajectories
    
    def _select_diverse_trajectories(self, trajectories, field):
        """Select diverse subset using max-min Hausdorff distance"""
        if len(trajectories) == 0:
            # Fallback: generate straight lines
            trajs = []
            for i in range(self.num_final_trajectories):
                angle = 2 * np.pi * i / self.num_final_trajectories
                end_point = np.array([np.cos(angle), np.sin(angle)]) * 3.0
                traj = np.linspace([0, 0], end_point, self.trajectory_length)
                trajs.append(traj)
            
            trajs_tensor = torch.tensor(trajs, dtype=torch.float32, device=field.device)
            weights = torch.ones(self.num_final_trajectories, device=field.device)
            weights = F.softmax(weights / self.temperature, dim=0)
            return trajs_tensor, weights
        
        # Score trajectories
        scores = []
        for traj in trajectories:
            # Average success probability along trajectory
            traj_grid = self._coords_to_grid_indices(traj)
            valid_mask = (traj_grid[:, 0] >= 0) & (traj_grid[:, 0] < self.grid_size) & \
                        (traj_grid[:, 1] >= 0) & (traj_grid[:, 1] < self.grid_size)
            
            if valid_mask.sum() > 0:
                field_values = field[traj_grid[valid_mask, 0], traj_grid[valid_mask, 1]]
                avg_prob = field_values.mean().item()
            else:
                avg_prob = 0.0
            
            # Length penalty
            length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            
            # Curvature penalty
            if len(traj) > 2:
                vel = np.diff(traj, axis=0)
                acc = np.diff(vel, axis=0)
                curvature = np.mean(np.linalg.norm(acc, axis=1))
            else:
                curvature = 0.0
            
            score = avg_prob - 0.1 * length - 0.05 * curvature
            scores.append(score)
        
        # Greedy selection for diversity
        selected = []
        remaining = list(range(len(trajectories)))
        
        # Select best scoring trajectory first
        best_idx = remaining[np.argmax([scores[i] for i in remaining])]
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Select remaining trajectories based on max-min distance
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
        
        # Pad if necessary
        while len(selected) < self.num_final_trajectories:
            selected.append(selected[-1])
        
        # Convert to tensors
        final_trajs = torch.tensor(
            np.array([trajectories[i] for i in selected[:self.num_final_trajectories]]),
            dtype=torch.float32,
            device=field.device
        )
        
        # Compute mixture weights
        selected_scores = torch.tensor(
            [scores[i] for i in selected[:self.num_final_trajectories]],
            dtype=torch.float32,
            device=field.device
        )
        weights = F.softmax(selected_scores / self.temperature, dim=0)
        
        return final_trajs, weights
    
    def _coords_to_grid_indices(self, coords):
        """Convert workspace coordinates to grid indices"""
        indices = []
        for x, y in coords:
            i = int((y - self.workspace_bounds[0]) / 
                   (self.workspace_bounds[1] - self.workspace_bounds[0]) * self.grid_size)
            j = int((x - self.workspace_bounds[0]) / 
                   (self.workspace_bounds[1] - self.workspace_bounds[0]) * self.grid_size)
            indices.append([i, j])
        return np.array(indices)
    
    def compute_loss(self, field_pred, expert_paths, coords):
        """Training loss for the success field
        
        Args:
            field_pred: [B, N] predicted success probabilities
            expert_paths: List of expert trajectory coordinates
            coords: [B, N, 2] coordinates corresponding to field_pred
        """
        B, N = field_pred.shape
        
        # Create labels
        labels = torch.zeros_like(field_pred)
        
        for b in range(B):
            if b < len(expert_paths):
                for point in expert_paths[b]:
                    # Find nearest grid points
                    dists = torch.norm(coords[b] - point.unsqueeze(0), dim=-1)
                    nearby = dists < 0.5  # Within 0.5m
                    labels[b, nearby] = 1.0
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(field_pred, labels)
        
        # Laplacian regularization for smoothness
        field_grid = field_pred.reshape(B, self.grid_size, self.grid_size)
        laplacian = torch.zeros_like(field_grid)
        laplacian[:, 1:-1, 1:-1] = (
            field_grid[:, :-2, 1:-1] + field_grid[:, 2:, 1:-1] +
            field_grid[:, 1:-1, :-2] + field_grid[:, 1:-1, 2:] -
            4 * field_grid[:, 1:-1, 1:-1]
        )
        smooth_loss = (laplacian ** 2).mean()
        
        return bce_loss + 0.01 * smooth_loss


# Example usage
if __name__ == "__main__":
    # Initialize module
    model = Field2Prior(
        context_dim=256,  # Assuming T=256 in [B, T]
        num_final_trajectories=3,
        trajectory_length=20
    )
    
    # Example input
    batch_size = 4
    temporal_dim = 256
    features = torch.randn(batch_size, temporal_dim)
    
    # Forward pass
    trajectories, weights = model(features)
    
    print(f"Output trajectories shape: {trajectories.shape}")  # [4, 3, 20, 2]
    print(f"Output weights shape: {weights.shape}")  # [4, 3]
    print(f"Weights sum per batch: {weights.sum(dim=1)}")  # Should be close to 1.0