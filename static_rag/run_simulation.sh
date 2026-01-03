#!/bin/bash
#SBATCH -J sim_rag_b
#SBATCH --gres=gpu:1
#SBATCH -p batch_ugrad                
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/sim_rag_b_%A.out
#SBATCH -e logs/sim_rag_b_%A.err

# 작업 정보 출력
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# 로그 디렉토리 생성
mkdir -p logs

# Conda 환경 활성화
source /data/mcladinz/anaconda3/etc/profile.d/conda.sh
conda activate base 

# GPU 확인
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# PYTHONPATH 설정 (프로젝트 루트)
export PYTHONPATH=$PYTHONPATH:.

# 시뮬레이션 실행
echo "=========================================="
echo "Starting Static RAG Simulation (Model B)..."
echo "=========================================="

# 주의: simulation_model_b.py 내부에서 Ollama를 사용하도록 설정되어 있다면,
# 이 스크립트 실행 전에 해당 노드에서 Ollama 서버가 실행 중이어야 합니다.
# 혹은 OpenAI API를 사용한다면 OPENAI_API_KEY가 .env에 있어야 합니다.

python static_rag/simulation_model_b.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Simulation completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Simulation failed!"
    echo "=========================================="
    exit 1
fi

# 최종 결과 요약
echo ""
echo "Job Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo "Logs: logs/sim_rag_b_${SLURM_JOB_ID}.out"
echo "Result: static_rag/Team2_StaticRAG_Results.csv"
