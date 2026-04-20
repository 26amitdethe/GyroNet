"""Generate the shared logAge grid bundled with the package.

500 points from logAge=0.0 (1 Myr) to logAge=4.14 (~13,800 Myr).
Resolution: ~0.0083 dex per bin, equivalent to ~2% relative age precision,
which is well below any model's actual posterior width.
"""

import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "gyronet" / "data" / "logA_grid.npy"

def main():
    grid = np.linspace(0.0, 4.14, 500)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, grid)
    print(f"Saved {len(grid)}-point logAge grid to {OUTPUT}")
    print(f"  logAge range: [{grid[0]}, {grid[-1]}]")
    print(f"  Age range (Myr): [{10**grid[0]:.2f}, {10**grid[-1]:.1f}]")
    print(f"  Spacing: {grid[1] - grid[0]:.5f} dex")

if __name__ == "__main__":
    main()
