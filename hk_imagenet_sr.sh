INPUT_ROOT="/tmp2/ICML2025"
EXP="/tmp2/ICML2025/ddnm"
IMAGE_FOLDER="/tmp2/ICML2025/ddrm/imagenet"
export CUDA_VISIBLE_DEVICES=0

python main.py --exp $EXP --ni --config imagenet_256.yml --doc imagenet --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 5
python main.py --exp $EXP --ni --config imagenet_256.yml --doc imagenet --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 10
python main.py --exp $EXP --ni --config imagenet_256.yml --doc imagenet --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 20
