from imports import *

@dataclass
class TrainingArgs():

    seed: int = 1
    lr: float = 1e-5
    batch_size: int = 12
    num_workers: int = os.cpu_count()
    max_prune_epochs: str = 5
    max_train_epochs: str = 3
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = None

    architecture: str = "deepset/roberta-base-squad2"
    prune_type: str = "l1"
    prune_rate: float = 0.2
    prune_ff_only: bool = True
    
    root_dir: str = './'
    checkpoint: str = './checkpoints'
    experiment: str = f"pruning_{prune_rate}"