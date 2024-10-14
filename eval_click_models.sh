NPROC_PER_NODE=$1

echo "Evaluate GrabCut"
torchrun --nproc-per-node $NPROC_PER_NODE scripts/evaluate_click_models.py -c 0 -d GrabCut -o experiments/eval_cm
echo "Evaluate Berkeley"
torchrun --nproc-per-node $NPROC_PER_NODE scripts/evaluate_click_models.py -c 0 -d Berkeley -o experiments/eval_cm
echo "Evaluate DAVIS"
torchrun --nproc-per-node $NPROC_PER_NODE scripts/evaluate_click_models.py -c 0 -d DAVIS -o experiments/eval_cm
echo "Evaluate COCO"
torchrun --nproc-per-node $NPROC_PER_NODE scripts/evaluate_click_models.py -c 0 -d COCO -o experiments/eval_cm
echo "Evaluate TETRIS"
torchrun --nproc-per-node $NPROC_PER_NODE --rdzv_endpoint=localhost:29400 scripts/evaluate_click_models.py -c 0 -s cm_ablation_checkpoints -d TETRIS -o experiments/eval_cm