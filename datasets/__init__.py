from .dataset import CommDataset as DS
from .dataset_multi import MultiDataset as DS_multi
from .dataset_mds import MultiDataset as DS_mds
from .dataset_mt import MTDataset as DS_mt

__all__ = ("DS", "DS_mt", "DS_multi", "DS_mds")