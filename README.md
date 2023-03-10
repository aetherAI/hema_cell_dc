# hema_cell_dc
This repository provides scripts to reproduce the results in the paper "A Machine-Learning-Based Algorithm for Bone Marrow Cell
Differential Counting".


## License
Copyright (C) 2021 aetherAI Co., Ltd.
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Requirements
```
pip install -r requirements.txt
```

## Training
To train the hema dc model, simply run
```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/config.py
```
The model will save in `'./work_dirs/$CONFIG_NAME$/'`

## Testing
After train the model, the predictions on the test-set can be generated by
```bash
CUDA_VISIBLE_DEVICES=0 python test.py ./work_dirs/$CONFIG_NAME/$CONFIG_NAME.py ./work_dirs/$CONFIG_NAME/epoch_12.pth --out ./work_dirs/$CONFIG_NAME/prediction.pkl
```