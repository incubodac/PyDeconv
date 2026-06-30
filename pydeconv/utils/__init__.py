# Utility subpackage exports

# 1. Artifact & Window Rejection
from .window_rejection import (
    cont_ArtifactDetect,
    basicrap,
    joinclosesegments
)

# 2. Statistics & Group Analysis (TFCE)
from .tfce import (
    tfce,
    get_channel_adjacency
)

# 3. Design Matrix (Placeholder for exports when implemented)
from .design_matrix import create_design_matrix 

# 4. Metrics (Placeholder for exports when implemented)
# from .metrics import calculate_vif, calculate_aic, calculate_pearson_r

# 5. Plotting (Placeholder for exports when implemented)
# from .plotting import plot_coefficients, plot_design_matrix

# 6. Event Stats (Placeholder for exports when implemented)
# from .event_stats import compute_event_stats

# 7. I/O helpers (Placeholder for exports when implemented)
# from .io import load_set_file

__all__ = [
    # Rejection
    'cont_ArtifactDetect',
    'basicrap',
    'joinclosesegments',
    # Stats / TFCE
    'tfce',
    'get_channel_adjacency',
]
