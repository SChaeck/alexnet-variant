#!/bin/bash

# 실행 권한 부여 후 실행 필요 #
# chmod +x run_code.sh
# ./run_code.sh

# Step 0: 프로젝트 루트 디렉토리 설정
PROJECT_DIR=$(dirname "$(realpath "$0")")  # 스크립트가 위치한 디렉토리를 기준으로 PROJECT_DIR 설정
cd "$PROJECT_DIR" || exit 1                # 디렉토리로 이동, 실패 시 스크립트 종료
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"  # 프로젝트 루트를 PYTHONPATH에 추가


# Step 1: 데이터 시각화
python main/data_analysis.py

# # Step 2: 모델별 학습 수행 (model, batch size, epoch, learning rate)
python main/training.py a 64 10 1e-4     # AlexNet 
python main/training.py r 64 10 1e-4     # Residual이 추가된 AlexNet
python main/training.py c 64 10 1e-4     # Linear Classifier로 분류하는 AlexNet

# Step 3: Execute test.py with different arguments
python main/test.py a          
python main/test.py r
python main/test.py c

# Step 4: Execute embedding_space_analysis.py
python main/embedding_space_analysis.py       # 임베딩 공간 분석

# Notify when the script is complete
echo "완료"
