import os
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data import data_load
from model import AlexNet, AlexNetWithResidual, LinearRegressor, AlexNetWithClassifier

# 실행 시 입력된 인자 처리
if len(sys.argv) < 2:
    print("Usage: python script.py [a|r]")
    print("a: Test AlexNet")
    print("r: Test AlexNetWithResidual")
    print("c: Test AlexNetWithClassifier")
    sys.exit(1)

# 모델 선택 인자
model_type = sys.argv[1].lower()
if model_type not in ['a', 'r', 'c']:
    print("Invalid argument. Use 'a' for AlexNet \nor 'r' for AlexNetWithResidual \nor 'c' for AlexNetWithClassifier")
    sys.exit(1)

# device 설정
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 로더
train_dataset, train_loader, test_dataset, test_loader, classes = data_load('cifar-10', 64)

# 모델 설정
if model_type == 'a':
    print("Using AlexNet model.")
    model = AlexNet(num_classes=10, dropout=0.2).to(device)
    model.load_state_dict(torch.load('./outputs/alexnet/best_model/model.pth'))
    file_suffix = "alexnet"
    
elif model_type == 'r':
    print("Using AlexNetWithResidual model.")
    model = AlexNetWithResidual(num_classes=10, dropout=0.2).to(device)
    model.load_state_dict(torch.load('./outputs/alexnet_with_residual/best_model/model.pth'))
    file_suffix = "alexnet_with_residual"

elif model_type == 'c':
    print("Using AlexNetWithClassifier model.")

    # 모델 정의 및 로드
    alexnet = AlexNet(num_classes=10, dropout=0.2).to(device)
    classifier = LinearRegressor(4096, 10).to(device)
    model = AlexNetWithClassifier(alexnet, classifier)
    model.load_state_dict(torch.load('./outputs/alexnet_with_classifier/best_model/model.pth'))

    file_suffix = "alexnet_with_classifier"

# Confusion Matrix
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        o, _ = model(x)
        _, pred = o.max(1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
save_path = f'outputs/pictures/2_confusion_matrix_{file_suffix}'

# 디렉토리 생성
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)  # Create intermediate directories


plt.savefig(save_path)
print(f"Image saved to {save_path}")
