# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class BDD100kDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(BDD100kDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_train_id.png',
            **kwargs)
    