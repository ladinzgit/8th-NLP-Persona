#!/usr/bin/bash
#SBATCH -J Model1_Sim      # Job 이름 (Model1_Sim으로 설정)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v7     
#SBATCH -t 1-0
#SBATCH -o /data/a2024105535/khuda_pro/static_zero_shot/logs/slurm-%A.out
#SBATCH -e /data/a2024105535/khuda_pro/static_zero_shot/logs/slurm-%A.err

# 시작 시간 기록
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# 1. 작업 디렉토리로 이동 (절대 경로 사용)
cd /data/a2024105535/khuda_pro/static_zero_shot
echo "Moved to Working directory: $(pwd)"

# 2. 가상환경 활성화 (prompt_lab)
# .bashrc를 로드하여 conda 명령어를 사용할 수 있게 함
source ~/.bashrc
conda activate prompt_lab

# 현재 파이썬 경로 확인 (환경이 잘 켜졌는지 확인용)
echo "Python path being used:"
which python

# 3. 실행 (Model 1 시뮬레이션)
python simulation_model_a.py

# 종료 시간 기록
echo "Job finished at: $(date)"

