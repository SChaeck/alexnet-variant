import os
import sys
import torch

from data import data_load
from model import AlexNet, AlexNetWithResidual, LinearRegressor, AlexNetWithClassifier, CustomTrainer

# 실행 시 입력된 인자 처리
if len(sys.argv) < 2:
    print("Usage: python script.py [a|r]")
    print("a: Run AlexNet")
    print("r: Run AlexNetWithResidual")
    print("c: Run AlexNetWithClassifier")
    sys.exit(1)

# 모델 선택 인자
model_type = sys.argv[1].lower()
if model_type not in ['a', 'r', 'c']:
    print("Invalid argument. Use 'a' for AlexNet \nor 'r' for AlexNetWithResidual \nor 'c' for AlexNetWithClassifier")
    sys.exit(1)

# device, 하이퍼파라미터 설정
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
epoch = int(sys.argv[3]) if len(sys.argv) > 3 else 50
learning_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4

# 데이터 로더
train_dataset, train_loader, test_dataset, test_loader, classes = data_load('cifar-10', batch_size)

print()

# 모델 설정
if model_type == 'a':
    print("Using AlexNet model.")
    model = AlexNet(num_classes=10, dropout=0.2).to(device)
    save_dir = 'outputs/alexnet'
    
elif model_type == 'r':
    print("Using AlexNetWithResidual model.")
    model = AlexNetWithResidual(num_classes=10, dropout=0.2).to(device)
    save_dir = 'outputs/alexnet_with_residual'
    
elif model_type == 'c':
    print("Using AlexNetWithClassifier model.")

    # 사전 학습된 AlexNet이 없는 경우 가이드
    if not os.path.exists(f'./outputs/alexnet/best_model'):
        print("You must train AlexNet first")   
        sys.exit(1) 

    # AlexNet 로드, 파라미터 고정
    alexnet = AlexNet(num_classes=10, dropout=0.2).to(device)
    alexnet.load_state_dict(torch.load('./outputs/alexnet/best_model/model.pth'))
    for param in alexnet.parameters():
        param.requires_grad = False
        
    # Linear Regressor 정의
    classifier = LinearRegressor(4096, 10).to(device)
    
    # 모델 정의
    model = AlexNetWithClassifier(alexnet, classifier)

    save_dir = 'outputs/alexnet_with_classifier'

# 트레이너 설정
trainer = CustomTrainer(
    model=model,                
    train_loader=train_loader,
    test_loader=test_loader,
    batch_size=batch_size,
    epoch=epoch,
    learning_rate=learning_rate,
    criterion='cross-entropy',
    optimizer='adam',
    logging_epoch=1,
    eval_epoch=1,
    save_epoch=1,
    save_dir=save_dir,
    save_best_model=True,
    device=device
)

# 훈련
trainer.fit()