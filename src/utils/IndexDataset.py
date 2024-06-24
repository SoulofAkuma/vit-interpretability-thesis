from torch.utils.data.dataset import Dataset
from src.utils.imagenet import MAPPING_FRAME
from typing import List

class IndexDataset(Dataset):
    """A simplistic class to include the index in the 

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        
    def __getitem__(self, index):
        return {'dataset_index': index}
    
    def get_imagenet_classes(self) -> List[str]:
        return MAPPING_FRAME.index.tolist()