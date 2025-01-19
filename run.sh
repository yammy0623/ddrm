EXP="/tmp2/ICML2025/ddrm_org"
INPUT_ROOT="/tmp2/ICML2025"
export CUDA_VISIBLE_DEVICES=0
python main.py --exp $EXP --ni --config imagenet_256.yml --doc imagenet --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 --input_root $INPUT_ROOT
