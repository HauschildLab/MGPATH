"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
import pandas
from torch.utils.data import Dataset

class DatasetHandler(Dataset):
    def __init__(
        self,
        seed: int,
        csv_path: str,
        label_dict: dict,
    ):
        self.seed = seed
        self.label_dict = label_dict
        self.csv_path = csv_path
        self.num_classes = len(set(self.label_dict.values()))
        self.slide_ids, self.labels = self.load_slide_ids(csv_path)

    def __getitem__(
        self, 
        index: int,
    ) -> None:
        pass

    def load_slide_ids(
        self,
        csv_path: str,
    ) -> tuple[list, list]:
        """
        @desc:
            1) load slide ids from the label csv file.
            2) return a list of slide ids and a list of labels
            3) the label csv file must have the following columns:
                - slide_id
                - label
        """
        slide_df = pandas.read_csv(csv_path)
        slide_ids = slide_df['slide_id'].tolist()
        labels = slide_df['label'].tolist()
        return slide_ids, labels

    def __len__(
        self
    ) -> int:
        return len(self.slide_ids)
