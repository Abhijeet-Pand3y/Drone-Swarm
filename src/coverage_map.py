import torch
import torch.nn.functional as F


class CoverageMap:
    """
    Tracks scanned cells across all parallel envs.
    
    Owns: grid resolution, cell marking, map queries.
    Does NOT own: scan range (drone property), encoding strategy (observation.py).
    
    Phase 0: get_full_map() used in observation
    Phase 2+: get_local_window() + hierarchical encoding replaces get_full_map()
    """
    def __init__(self, num_envs: int, arena_size: float, cell_size: float, device: str):
        self.num_envs = num_envs
        self.arena_size = arena_size
        self.cell_size = cell_size
        self.device = device

        self.grid_size = int(arena_size / cell_size)
        # float scan-progress grid, 0.0 = untouched, 1.0 = fully scanned
        self.grid = torch.zeros(
            (self.num_envs, self.grid_size, self.grid_size),
            dtype=torch.float, device=self.device
        )
        self.cell_centers = self._compute_cell_centers()

        # per-agent increment buffer is  float 
        self._batch_progress = torch.zeros(
            (self.num_envs, self.grid_size, self.grid_size),
            dtype=torch.float, device=self.device
        )

    def _compute_cell_centers(self) -> torch.Tensor:
        """
        Precompute the (x, y) world coordinates of each cell center for efficient queries.
        Returns:
            (grid_size, grid_size, 2) tensor of (x, y) coords for each cell center
        """
        half_cell = self.cell_size / 2
        
        x_coords = torch.arange(half_cell, self.arena_size, self.cell_size, device=self.device)
        y_coords = torch.arange(half_cell, self.arena_size, self.cell_size, device=self.device)

        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

        return torch.stack((grid_x, grid_y), dim=-1)  # shape: (grid_size, grid_size, 2)
    
    
    def reset(self, env_ids: torch.Tensor):
        self.grid[env_ids] = 0.0

    def pos_to_cell(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous (x, y) world positions to (cell_x, cell_y) indices.

        Args:
            positions: (num_envs, 2) float tensor of (x, y) world coords
        Returns:
            (num_envs, 2) long tensor of (cell_x, cell_y), clamped to valid range
        """
        pos_xy = positions[:, :2]
        cell_indices = torch.floor(pos_xy / self.cell_size).long()
        return cell_indices.clamp(0, self.grid_size - 1)
    
    def mark_scanned(self, env_ids: torch.Tensor, positions: torch.Tensor,
                 scan_range: float, scan_rate: float = 0.2, threshold=0.95):
        """
        Accumulate scan progress for cells within scan_range of each agent.
        Cells gain `scan_rate` progress per step of presence, clamped at 1.0.

        Multi-agent same cell: progress increment is the MAX across agents in that
        env (presence-based — two drones over one cell don't scan it 2x faster;
        coordinated co-scanning is out of scope). That max increment is added to
        the running grid and clamped.

        Args:
            env_ids:    (num_agents_total,) — env index per agent
            positions:  (num_agents_total, 2 or 3) — agent world positions
            scan_range: float — physical scan radius in meters
            scan_rate:  float — progress added per step of presence (default 0.2)
        Returns:
        progress_delta:   (num_envs,) total scan-progress added this step (clamp-accounted)
        completion_count: (num_envs,) cells that newly reached `threshold` this step
        """
        pos_xy = positions[:, :2]
        pos_expanded = pos_xy[:, None, None, :]
        diff = self.cell_centers - pos_expanded
        dist_sq = (diff ** 2).sum(dim=-1)
        within_range = dist_sq <= (scan_range ** 2)
        increments = within_range.float() * scan_rate

        self._batch_progress.zero_()
        self._batch_progress.index_reduce_(0, env_ids, increments, reduce="amax", include_self=True)

        # snapshot for delta accounting
        grid_before = self.grid.clone()
        completed_before = (grid_before >= threshold)

        self.grid = (self.grid + self._batch_progress).clamp_(max=1.0)

        # clamp-accounted progress actually added, per env
        progress_delta = (self.grid - grid_before).sum(dim=(1, 2))          # (num_envs,)
        # cells that newly crossed the threshold this step, per env
        completed_after = (self.grid >= threshold)
        newly_completed = completed_after & (~completed_before)
        completion_count = newly_completed.sum(dim=(1, 2)).float()          # (num_envs,)

        return progress_delta, completion_count


    def get_full_map(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Return full coverage map for each agent's environment.
        
        Args:
            env_ids: (num_agents_total,) — env index for each agent
        Returns:
            (num_agents_total, grid_size * grid_size) float tensor
        """
        # flatten grid to (num_envs, grid*grid), then index by env_ids
        flat = self.grid.float().view(self.num_envs, -1)   # (num_envs, grid*grid)
        return flat[env_ids]                                # (num_agents_total, grid*grid)
    
    def get_local_window(self, env_ids: torch.Tensor, positions: torch.Tensor, local_range: float, output_size: int = 11) -> torch.Tensor:
        """
        Extract fixed-size local coverage windows centered on each individual agent.
        
        This is a Per-Agent method. It operates on a flattened batch layout to allow
        multiple agents within the same environment to query their shared grid slice.

        Args:
            env_ids:     (num_envs * num_agents,) Long tensor mapping each agent row to its env index
            positions:   (num_envs * num_agents, 2) or (..., 3) Float tensor of flat world coordinates
            local_range: float, physical sensor radius in meters
            output_size: int, the fixed square resolution of the output grid (default 11x11)
        Returns:
            (num_envs * num_agents, output_size * output_size) Float tensor. 
            0.0 = unscanned, 1.0 = scanned. Out-of-bounds padded with 1.0.
        """
        num_agents_total = positions.shape[0]
        
        # 1. Map continuous coordinates to grid indices
        center_cells = self.pos_to_cell(positions)          
        window_half = int(local_range / self.cell_size)   

        # 2. Pad master grid boundaries so edge agents always get a complete window slice
        padded = F.pad(
            self.grid.float(),
            pad=(window_half, window_half, window_half, window_half),
            value=1.0
        )

        # 3. Shift local coordinate targets into padded matrix space
        cx = center_cells[:, 0] + window_half   
        cy = center_cells[:, 1] + window_half   

        # 4. Generate coordinate index meshgrid
        offsets = torch.arange(-window_half, window_half + 1, device=self.device)
        dx, dy = torch.meshgrid(offsets, offsets, indexing='ij')

        # Broadcast agent centers against localized offsets to extract window patches
        global_x = cx.view(-1, 1, 1) + dx   
        global_y = cy.view(-1, 1, 1) + dy   

        # 5. Extract localized windows across environments simultaneously
        env_idx = env_ids.view(-1, 1, 1) 
        raw_windows = padded[env_idx, global_x, global_y]  # Shape: (num_agents_total, wl, wl)

        # 6. Downsample patches to locked observation size via adaptive average pooling
        raw_windows = raw_windows.unsqueeze(1) # Shape: (num_agents_total, 1, wl, wl)
        pooled = F.adaptive_avg_pool2d(raw_windows, (output_size, output_size))
        
        # Remove the explicit channel dimension safely before reshaping
        pooled = pooled.squeeze(1) # Shape: (num_agents_total, output_size, output_size)

        return pooled.reshape(num_agents_total, -1) 
    
    def get_coverage_pct(self, env_ids: torch.Tensor | None = None,
                     threshold: float = 0.95) -> torch.Tensor:
        """
        Fraction of cells considered COVERED (progress >= threshold).

        Args:
            env_ids:   (num_agents_total,) for per-agent, or None for per-env
            threshold: progress level counting a cell as covered (default 0.95)
        Returns:
            per-agent (num_agents_total,) or per-env (num_envs,)
        """
        total_cells = self.grid_size * self.grid_size
        covered = (self.grid >= threshold).float()                  # CHANGED: threshold, not nonzero
        per_env = covered.sum(dim=(1, 2)) / total_cells

        if env_ids is None:
            return per_env
        return per_env[env_ids]

    def is_fully_covered(self, env_ids: torch.Tensor | None = None,
                     threshold: float = 0.95) -> torch.Tensor:
        """Check whether coverage (cells at/above threshold) meets the threshold fraction."""
        return self.get_coverage_pct(env_ids, threshold=threshold) >= threshold
        
