# RClicks: Realistic Click Simulation for Benchmarking Interactive Segmentation

[![Page](https://img.shields.io/badge/Project-Page-blue)](https://emb-ai.github.io/rclicks-project)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2410.11722)  
This repository provides the code to estimate **performance** and **robustness** of click-based interactive segmentation methods w.r.t. clicks positions.

<img alt="image" src="https://github.com/user-attachments/assets/27273792-5856-4e18-8d1f-46c25f3941a4">

# Setting up an environment
## Install dependencies:

```cmd
# create conda env
conda create -n rclicks python==3.9.19
conda activate rclicks
# install python packages
pip install -r requirements.txt

# install mmsegmentation and transalnet
mim install mmcls==0.25.0 mmcv-full==1.7.2 mmengine==0.10.2
pip install -e mmsegmentation
pip install -e transalnet

# install our packages
pip install -e isegm
pip install -e rclicks

# For benchmarking Segment-Anything (SAM), SAM-HQ, MobileSAM, SAM2/2.1
# Please install it separately e.g:
# git clone https://github.com/facebookresearch/segment-anything.git
# pip install -e segment-anything

```

## Prepare datasets & models checkpoints
This project is mostly developed based on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation) and use the same dataset structure and evaluation scripts. Thus, you should configure the paths to the datasets in config.yml. However, currently all paths for `rclicks` package are hard-coded in the `rclicks/rclicks/paths.py`, we will change it in the next release.

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|Grab Cut   |  50 images with one object each (test)       |  [GrabCut.zip (11 MB)][GrabCut]      |
|Berkeley   |  96 images with 100 instances (test)         |  [Berkeley.zip (7 MB)][Berkeley]     |
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|COCO_MVal  |  800 images with 800 instances (test)        |  [COCO_MVal.zip (127 MB)][COCO_MVal] |
|TETRIS     |  2000 images with 2531 instances (test)      |  [TETRIS.zip (6.3 GB)][TETRIS]       |
|PREVIEWS (TETRIS)     |  100 images and masks from TETRIS used to ablate display modes    |  [PREVIEWS.zip (298 MB)][PREVIEWS]       |
|SUBSEC_MASKS     |  RClicks masks for subsequent clicks    |  [SUBSEC_MASKS.zip (36 MB)][SUBSEC_MASKS]       |

[GrabCut]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
[Berkeley]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[COCO_MVal]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip
[TETRIS]: https://drive.google.com/file/d/1iJgohY1XBSnY-kRUoaRZJlu0HZbyHcyK/view?usp=sharing  
[PREVIEWS]: https://drive.google.com/drive/folders/1mcCL0lccassABzt1Tm16gYPO3-V5O3uf?usp=sharing
[SUBSEC_MASKS]: https://drive.google.com/drive/folders/17NbsWTpfA_WyUirTf55LsC90keN2W-Kg?usp=sharing

Please download all datasets and place them into `datasets` directory.

Checkpoints for a saliency model and our **clickability model** can be downloaded here [CLICKABILITY_CHECKPOINTS.zip (445 MB)](https://drive.google.com/drive/folders/1ERy2swCS35qT_J0B-QQsPCfyzXbPODK7?usp=sharing). Please unzip it right into project directory. Make sure that `clickability_model.pth` is located in the root of the project directory; and `resnet50-0676ba61.pth` and `TranSalNet_Res.pth` are located in `transalnet\transalnet\pretrained_models`.

To download interactive segmentation methods checkpoints, please refer to the repositories of the relevant papers or download all checkpoints used in this work at once — [MODELS_CHECKPONTS.zip (21.5 GB)](https://drive.google.com/file/d/1lVP2u5wYqE72S-9Mhw_Y3EGGlJmq5uOg/view?usp=sharing)

# Run optimization
## Clicking Groups Sampling Example
```
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=coco_lvis_h18_itermask.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_count=1 --trajectory_selection=-1 --trajectory_sampling_prob_low=0.9 --trajectory_sampling_prob_high=1.0
```

All flags the same as in original models except following additional flags:  
```  
--n_workers — number of parallel workers for evaluation (the maximum number you can fit depends on your GPU)
--clickability_model_pth — path to clickability_model checkpoint
--trajectory_sampling_count — we used one sample
--trajectory_sampling_prob_low — lower bound to slice probability mass
--trajectory_sampling_prob_high — upper bound to slice probability mass
```  


## Real-User Evaluation Example
```
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=coco_lvis_h18_itermask.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=1 --n_workers=1 --iou-analysis --thresh=0.5 --user_inputs 
```

All flags the same as in original models except following additional flags:  
```  
--n_workers — number of parallel workers for evaluation (the maximum number you can fit depends on your GPU)
--user_inputs — user clicks benchmarking
```  


## Full Benchmarking
Some models ([SAM](https://github.com/facebookresearch/segment-anything), [SAM-HQ](https://github.com/SysCV/sam-hq), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [SAM2/2.1](https://github.com/facebookresearch/sam2/)) should be installed using separate package.

To benchmark all models ([SAM](https://github.com/facebookresearch/segment-anything), [SAM2/2.1](https://github.com/facebookresearch/sam2/), [SAM-HQ](https://github.com/SysCV/sam-hq), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation/tree/master), [SimpleClick](https://github.com/uncbiag/SimpleClick/), [GPCIS](https://github.com/zmhhmz/GPCIS_CVPR2023), [CDNet](https://github.com/XavierCHEN34/ClickSEG), [CFR-ICL](https://github.com/TitorX/CFR-ICL-Interactive-Segmentation/)) after setting up an environment and downloading all checkpoints to ```MODEL_CHECKPOINTS``` folder and just run:

```cmd
bash run_clicking_groups.sh 0.0 0.1
bash run_clicking_groups.sh 0.1 0.2
bash run_clicking_groups.sh 0.2 0.3
bash run_clicking_groups.sh 0.3 0.4
bash run_clicking_groups.sh 0.4 0.5
bash run_clicking_groups.sh 0.5 0.6
bash run_clicking_groups.sh 0.6 0.7
bash run_clicking_groups.sh 0.7 0.8
bash run_clicking_groups.sh 0.8 0.9
bash run_clicking_groups.sh 0.9 1.0
bash run_base_user_sample.sh
```

All hyperparameters are set following author implementations.

We provide all obtained [evaluation_results.zip (896 MB)](https://drive.google.com/file/d/1IroXiDsbuECTt2z8GRZgaBxUGShBdE9y/view?usp=sharing) folders and `compute_benchmark_metrics.ipynb` to reproduce all benchmark metrics from the main paper and supplementary.


# Clicks and clickability model

For demonstration of usage `rclicks` package please refer to `rclicks_demo.ipynb`.

## Display modes ablations

To obtain display modes ablation results use the following command:
```cmd
python scripts/evaluate_previews.py
```
Results will be printed.

## PC vs mobile

To obtain comparison results between PC and mobile clicks run:
```cmd
python scripts/evaluate_mobile_pc.py
```
Results will be printed and saved into `experiments/pc_vs_mobile.csv`.

## Training clickability models

To train clickability model use the following command:
```cmd
python scripts/train_click_model.py --sigma 5
```
where `sigma` is hyperparameter (see paper).
Checkpoints will be saved into `experiments/train/sigma={sigma}` directory.


## Evaluate clickability models


### From scratch

To evaluate and ablate clickability models for each `sigma` from scratch you need to download [cm_ablation_checkpoints.zip (1.6 GB)](https://drive.google.com/drive/folders/1ERy2swCS35qT_J0B-QQsPCfyzXbPODK7?usp=sharing). Please unzip all `clickability_model_*.pth` files into `cm_ablation_checkpoints` directory in the project root.

To run evaluation script to calculate all metrics from scratch per dataset and per image use the following command:
```cmd
bash eval_click_models.sh $NPROC_NUMBER
```
Results will be saved as `.csv` files in `experiments/eval_cm` directory. 
In our experiments we used `NPROC_NUMBER=40` and evaluated on 8 A100 GPUs.


### Tables from the paper

To process precalculated `.csv` files with per sample metrics into tables from the paper call:
```cmd
python scripts/prepare_click_models_tables.py
```
- `experiments/eval_cm/eval_cm_tetris.csv` -- evaluation of clickability models for TETRIS (Val) (Table 3 in main paper).
- `experiments/eval_cm/eval_cm_all.csv` -- evaluation of clickability models for all datasets (Table 6 in Appendix B.2).
- `experiments/eval_cm/ablation_cm_sigma_tetris.csv` -- `sigma`-parameter ablation of our clickability models on TETRIS (Val) (Table 7 in Appendix B.3).

## Citation

Please cite the paper if you find challenge materials useful for your research:

```
@inproceedings{antonov2024rclicks,
 author = {Antonov, Anton and Moskalenko, Andrey and Shepelev, Denis and Krapukhin, Alexander and Soshin, Konstantin and Konushin, Anton and Shakhuro, Vlad},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {127673--127710},
 publisher = {Curran Associates, Inc.},
 title = {RClicks: Realistic Click Simulation for Benchmarking Interactive Segmentation},
 volume = {37},
 year = {2024}
}
```
