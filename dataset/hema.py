import numpy as np
from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class HemaDataset(CocoDataset):

    CLASSES = (
        'Basophil', 'Blast', 'Eosinophils-and-precursors',
        'Erythroid', 'Histiocyte', 'Invalid', 'Lymphocyte',
        'Metamyelocyte', 'Mitosis', 'Monocyte-and-precursors',
        'Myelocyte', 'PMN', 'Plasma-cell', 'Proerythroblast',
        'Promyelocyte'
    )

    def __init__(self, *args, **kwargs):
        super(HemaDataset, self).__init__(*args, **kwargs)
