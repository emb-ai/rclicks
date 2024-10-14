if [ -z $1 ]
  then
    echo "No PROB_LOW argument supplied"
    exit
fi

if [ -z $2 ]
  then
    echo "No PROB_HIGH argument supplied"
    exit
fi

echo PROB_LOW=$1
echo PROB_HIGH=$2


python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18_itermask.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18_baseline.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18s_itermask.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h32_itermask.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/sbd_h18_itermask.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_base.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_base.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_xtiny.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_large.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_large.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_huge.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_huge.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_cfr.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/CFR-ICL/cocolvis_icl_vit_huge.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --cf-n=1 --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_adaptiveclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/AdaptiveClick/adaptiveclick_base448_cocolvis --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_adaptiveclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/AdaptiveClick/adaptiveclick_base448_sbd --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_cdnet.py CDNet --checkpoint=./MODEL_CHECKPOINTS/CDNet/cclvs_last_checkpoint.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_cdnet.py CDNet --checkpoint=./MODEL_CHECKPOINTS/CDNet/sbd_last_checkpoint.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_gpcis.py Baseline --checkpoint=./MODEL_CHECKPOINTS/GPCIS/GPCIS_Resnet50.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.49 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}


# Install MobileSAM before use
python3 scripts/evaluate_model_mobilesam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/MobileSAM/mobile_sam.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

# Install SAM before use
python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_l_0b3195.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_h_4b8939.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

# Install SAM-HQ before use
python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_b.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_l.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_h.pth --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}


# Install SAM2 before use
python3 scripts/evaluate_model_sam2.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2/sam2_hiera_tiny.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam2.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2/sam2_hiera_small.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam2.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2/sam2_hiera_base_plus.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam2.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2/sam2_hiera_large.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}



# Install SAM2.1 before use
python3 scripts/evaluate_model_sam2.1.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_tiny.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam2.1.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_small.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam2.1.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_base_plus.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}

python3 scripts/evaluate_model_sam2.1.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_large.pt --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS --n-clicks=20 --n_workers=1 --iou-analysis --thresh=0.5 --clickability_model_pth clickability_model.pth --trajectory_sampling_prob_low=${1} --trajectory_sampling_prob_high=${2}
