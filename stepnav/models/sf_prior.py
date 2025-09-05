"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2025/09/01 21:29:38
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
import heapq
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class SuccessFieldModule(nn.Module):
    """
    Success Field to Structured Multi-Modal Prior Module
    
    Input: [B, 1024] features
    Output: [B, 8, 2] 2D trajectories
    """
    
    def __init__(self, 
                 feature_dim: int = 1024,
                 hidden_dim: int = 256,
                 grid_size: int = 64,
                 workspace_range: float = 10.0,
                 num_trajectories: int = 8,
                 max_waypoints: int = 20,
                 trajectory_length: int = 8,
                 temperature: float = 1.0,
                 laplacian_weight: float = 0.01):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.workspace_range = workspace_range
        self.num_trajectories = num_trajectories
        self.max_waypoints = max_waypoints
        self.trajectory_length = trajectory_length
        self.temperature = temperature
        self.laplacian_weight = laplacian_weight
        
        # Context encoder - compress features to context vector
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)  # z_c dimension
        )
        
        # Success field MLP g_φ(x, z_c)
        self.success_field_mlp = nn.Sequential(
            nn.Linear(2 + 64, hidden_dim),  # 2D coordinate + context
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # output success probability
        )
        
        # Create coordinate grid
        self.register_buffer('coord_grid', self._create_coordinate_grid())
        
    def _create_coordinate_grid(self):
        """Create egocentric coordinate grid"""
        x = torch.linspace(-self.workspace_range/2, self.workspace_range/2, self.grid_size)
        y = torch.linspace(-self.workspace_range/2, self.workspace_range/2, self.grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1)  # [grid_size, grid_size, 2]
        return coords
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            features: [B, 1024] input features
        Returns:
            trajectories: [B, 8, 2] output trajectories
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Encode context vector z_c
        z_c = self.context_encoder(features)  # [B, 64]
        
        # Generate success field F(x) = σ(g_φ(x, z_c))
        success_fields = self._generate_success_fields(z_c)  # [B, grid_size, grid_size]
        
        # Convert success fields to structured prior trajectories
        trajectories = []
        for i in range(batch_size):
            traj = self._success_field_to_trajectories(success_fields[i].cpu().numpy())
            trajectories.append(traj)
        
        # Stack and convert to tensor
        trajectories = torch.tensor(np.stack(trajectories), dtype=torch.float32, device=device)
        
        return trajectories
    
    def _generate_success_fields(self, z_c: torch.Tensor) -> torch.Tensor:
        """Generate success probability fields"""
        batch_size = z_c.shape[0]
        device = z_c.device
        
        # Flatten coordinate grid for MLP input
        coords_flat = self.coord_grid.view(-1, 2)  # [grid_size^2, 2]
        
        # Expand context for all coordinates
        z_c_expanded = z_c.unsqueeze(1).expand(batch_size, coords_flat.shape[0], -1)  # [B, grid_size^2, 64]
        coords_expanded = coords_flat.unsqueeze(0).expand(batch_size, -1, -1)  # [B, grid_size^2, 2]
        
        # Concatenate coordinates and context
        mlp_input = torch.cat([coords_expanded, z_c_expanded], dim=-1)  # [B, grid_size^2, 66]
        
        # Apply success field MLP
        success_logits = self.success_field_mlp(mlp_input.view(-1, mlp_input.shape[-1]))  # [B*grid_size^2, 1]
        success_probs = torch.sigmoid(success_logits)  # Apply σ activation
        
        # Reshape to field format
        success_fields = success_probs.view(batch_size, self.grid_size, self.grid_size)
        
        return success_fields
    
    def _success_field_to_trajectories(self, success_field: np.ndarray) -> np.ndarray:
        """Convert success field to trajectory prior"""
        try:
            # Step 1: Non-maximum suppression to find waypoints
            waypoints = self._find_waypoints(success_field)
            
            if len(waypoints) < 2:
                # Fallback: create simple trajectories if no waypoints found
                return self._create_fallback_trajectories()
            
            # Step 2: Create energy field E(x) = -log(F(x) + δ)
            delta = 1e-6
            energy_field = -np.log(success_field + delta)
            
            # Step 3: Find diverse paths between waypoints
            paths = self._find_diverse_paths(energy_field, waypoints)
            
            if len(paths) == 0:
                return self._create_fallback_trajectories()
            
            # Step 4: Score and select top-M diverse trajectories
            trajectories = self._select_diverse_trajectories(paths, success_field)
            
            return trajectories
            
        except Exception as e:
            print(f"Error in trajectory generation: {e}")
            return self._create_fallback_trajectories()
    
    def _find_waypoints(self, success_field: np.ndarray) -> List[Tuple[int, int]]:
        """Find waypoints using non-maximum suppression"""
        # Apply non-maximum suppression
        local_maxima = maximum_filter(success_field, size=5) == success_field
        
        # Find peaks above threshold
        threshold = np.percentile(success_field, 70)  # Top 30% values
        peaks = np.where((local_maxima) & (success_field > threshold))
        
        waypoints = list(zip(peaks[0], peaks[1]))
        
        # Sort by success probability and limit number
        waypoints = sorted(waypoints, key=lambda p: success_field[p], reverse=True)
        waypoints = waypoints[:self.max_waypoints]
        
        return waypoints
    
    def _find_diverse_paths(self, energy_field: np.ndarray, waypoints: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Find diverse paths between waypoints using A* search"""
        paths = []
        
        # Try to find paths between different pairs of waypoints
        for i in range(min(len(waypoints), 10)):  # Limit iterations
            for j in range(i + 1, min(len(waypoints), 10)):
                start, goal = waypoints[i], waypoints[j]
                
                # Run A* with small random perturbations for diversity
                for _ in range(3):  # Multiple runs for diversity
                    path = self._astar_search(energy_field, start, goal)
                    if path and len(path) > 3:
                        paths.append(path)
                        
                    if len(paths) >= 20:  # Limit total paths
                        break
                if len(paths) >= 20:
                    break
            if len(paths) >= 20:
                break
        
        return paths
    
    def _astar_search(self, energy_field: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
        
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < energy_field.shape[0] and 0 <= ny < energy_field.shape[1]:
                    neighbors.append((nx, ny))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                # Add small random perturbation for diversity
                tentative_g = g_score[current] + energy_field[neighbor] + np.random.normal(0, 0.1)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def _select_diverse_trajectories(self, paths: List[List[Tuple[int, int]]], success_field: np.ndarray) -> np.ndarray:
        """Select diverse trajectories using greedy max-min Hausdorff selection"""
        if not paths:
            return self._create_fallback_trajectories()
        
        # Convert paths to smooth trajectories
        smooth_trajs = []
        for path in paths:
            if len(path) < 3:
                continue
                
            # Convert to world coordinates
            world_path = []
            for px, py in path:
                wx = (px / self.grid_size - 0.5) * self.workspace_range
                wy = (py / self.grid_size - 0.5) * self.workspace_range
                world_path.append([wx, wy])
            
            # Smooth with B-splines
            try:
                world_path = np.array(world_path)
                if len(world_path) >= 4:  # Need at least 4 points for cubic spline
                    tck, u = splprep([world_path[:, 0], world_path[:, 1]], s=0.1, k=3)
                    u_new = np.linspace(0, 1, self.trajectory_length)
                    smooth_x, smooth_y = splev(u_new, tck)
                    smooth_traj = np.column_stack([smooth_x, smooth_y])
                else:
                    # Linear interpolation for short paths
                    t = np.linspace(0, 1, self.trajectory_length)
                    smooth_traj = np.array([np.interp(t, np.linspace(0, 1, len(world_path)), world_path[:, i]) 
                                          for i in range(2)]).T
                
                smooth_trajs.append(smooth_traj)
            except:
                continue
        
        if not smooth_trajs:
            return self._create_fallback_trajectories()
        
        # Score trajectories
        scored_trajs = []
        for traj in smooth_trajs:
            score = self._score_trajectory(traj, success_field)
            scored_trajs.append((score, traj))
        
        # Sort by score
        scored_trajs.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy max-min Hausdorff selection for diversity
        selected = [scored_trajs[0][1]]  # Start with best trajectory
        
        for score, traj in scored_trajs[1:]:
            if len(selected) >= self.num_trajectories:
                break
            
            # Check diversity using Hausdorff distance
            min_dist = float('inf')
            for selected_traj in selected:
                try:
                    dist = max(directed_hausdorff(traj, selected_traj)[0],
                              directed_hausdorff(selected_traj, traj)[0])
                    min_dist = min(min_dist, dist)
                except:
                    min_dist = 0
            
            if min_dist > 1.0:  # Diversity threshold
                selected.append(traj)
        
        # Pad if not enough diverse trajectories
        while len(selected) < self.num_trajectories:
            selected.append(selected[0] + np.random.normal(0, 0.5, selected[0].shape))
        
        return np.array(selected[:self.num_trajectories])
    
    def _score_trajectory(self, traj: np.ndarray, success_field: np.ndarray) -> float:
        """Score trajectory based on success probability, length, and curvature"""
        # Convert world coordinates back to grid coordinates for field lookup
        grid_coords = []
        for point in traj:
            gx = int((point[0] / self.workspace_range + 0.5) * self.grid_size)
            gy = int((point[1] / self.workspace_range + 0.5) * self.grid_size)
            gx = np.clip(gx, 0, self.grid_size - 1)
            gy = np.clip(gy, 0, self.grid_size - 1)
            grid_coords.append((gx, gy))
        
        # Average success probability
        avg_success = np.mean([success_field[gx, gy] for gx, gy in grid_coords])
        
        # Path length (penalize very long paths)
        path_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        length_penalty = np.exp(-path_length / 5.0)
        
        # Curvature (penalize high curvature)
        curvature_penalty = 1.0
        if len(traj) > 2:
            # Simple curvature approximation
            diffs = np.diff(traj, axis=0)
            angles = np.diff(np.arctan2(diffs[:, 1], diffs[:, 0]))
            curvature = np.mean(np.abs(angles))
            curvature_penalty = np.exp(-curvature)
        
        return avg_success * length_penalty * curvature_penalty
    
    def _create_fallback_trajectories(self) -> np.ndarray:
        """Create fallback trajectories when path finding fails"""
        trajectories = []
        
        for i in range(self.num_trajectories):
            # Create simple forward trajectories with slight variations
            t = np.linspace(0, 1, self.trajectory_length)
            
            # Base trajectory: move forward
            x = t * 3.0 - 1.5  # Move 3 units forward, centered
            
            # Add variation for each trajectory
            angle = (i - self.num_trajectories/2) * 0.3  # Spread trajectories
            y = np.sin(angle) * t * 2.0
            
            trajectory = np.column_stack([x, y])
            trajectories.append(trajectory)
        
        return np.array(trajectories)
    
    def compute_laplacian_loss(self, success_field: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian regularization loss for smoothness"""
        # Compute second derivatives (Laplacian)
        # Using finite differences
        
        # Second derivative in x direction
        d2_dx2 = success_field[:, 2:, 1:-1] - 2 * success_field[:, 1:-1, 1:-1] + success_field[:, :-2, 1:-1]
        
        # Second derivative in y direction  
        d2_dy2 = success_field[:, 1:-1, 2:] - 2 * success_field[:, 1:-1, 1:-1] + success_field[:, 1:-1, :-2]
        
        # Laplacian = d2/dx2 + d2/dy2
        laplacian = d2_dx2 + d2_dy2
        
        # L2 norm of Laplacian
        return torch.mean(laplacian ** 2)


# Example usage and test
if __name__ == "__main__":
    # Create model
    model = SuccessFieldModule(
        feature_dim=1024,
        num_trajectories=8,
        trajectory_length=8
    )
    
    # Test forward pass
    batch_size = 4
    features = torch.randn(batch_size, 1024)
    
    with torch.no_grad():
        trajectories = model(features)
        print(f"Input shape: {features.shape}")
        print(f"Output shape: {trajectories.shape}")
        print(f"Output range: [{trajectories.min():.2f}, {trajectories.max():.2f}]")
        
        # Visualize one example
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        traj_sample = trajectories[0].numpy()  # First batch item
        
        for i, traj in enumerate(traj_sample):
            plt.plot(traj[:, 0], traj[:, 1], 'o-', label=f'Trajectory {i+1}', alpha=0.7)
            plt.arrow(traj[0, 0], traj[0, 1], 
                     traj[1, 0] - traj[0, 0], traj[1, 1] - traj[0, 1],
                     head_width=0.2, head_length=0.2, fc='red', ec='red')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate') 
        plt.title('Generated Multi-Modal Trajectory Prior')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()