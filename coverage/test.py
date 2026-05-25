import torch
from coverage_map import CoverageMap

cm = CoverageMap(num_envs=2, arena_size=50.0, cell_size=5.0, device="cuda:0")

# 3 agents: two in env 0 (overlapping), one in env 1
env_ids = torch.tensor([0, 0, 1], device="cuda:0")
positions = torch.tensor([
    [12.0, 12.0, 5.0],   # env 0, agent A
    [13.0, 13.0, 5.0],   # env 0, agent B — overlaps A
    [40.0, 40.0, 5.0],   # env 1, agent C
], device="cuda:0")

cm.mark_scanned(env_ids, positions, scan_range=5.0)

print("env 0 scanned cells:", cm.grid[0].sum().item())
print("env 1 scanned cells:", cm.grid[1].sum().item())
print("coverage per env:", cm.get_coverage_pct())
print("full map per agent shape:", cm.get_full_map(env_ids).shape)  # expect (3, 100)