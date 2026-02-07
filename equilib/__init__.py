from .topo_align import TopoAlignSolver
from .adaptive_topo_align import AdaptiveTopoAlignSolver
from .surrogate_topo_align import SurrogateTopoAlignSolver, NDimSurrogateTopoAlignSolver
from .ndim_topo_align import NDimTopoAlignSolver
from .sperner_trainer import SpernerTrainer

try:
    from . import plotting
except ImportError:
    plotting = None

__version__ = "0.1.0"
