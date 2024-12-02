import os
import torch
import matplotlib.pyplot as plt

from data import data_load, next_batch
from model import AlexNet, LinearRegressor, AlexNetWithClassifier, pca

# 디바이스, 하이퍼파라미터 설정
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_SIZE = 5000

# 데이터 로더
train_dataset, train_loader, test_dataset, test_loader, classes = data_load('cifar-10', 64)

plot_loader = torch.utils.data.DataLoader(test_dataset, batch_size=SAMPLE_SIZE, shuffle=False)
x, y = next_batch(plot_loader)
x = x.to(device)

# 모델 정의 및 로드
alexnet = AlexNet(num_classes=10, dropout=0.2).to(device)
classifier = LinearRegressor(4096, 10).to(device)
model = AlexNetWithClassifier(alexnet, classifier).to(device)
model.load_state_dict(torch.load('./outputs/alexnet_with_classifier/best_model/model.pth'))

pred, embedding = model(x)

data = embedding.detach().cpu().numpy()
labels = y.cpu().numpy()

transformed_data = pca(data)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    transformed_data[:, 0],
    transformed_data[:, 1],
    c=labels,
    cmap='tab20',
    alpha=0.7
)
plt.colorbar(scatter, label='Class Labels')
plt.title("Embedding Space")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
save_path = f'outputs/pictures/3_embedding_space'

# 디렉토리 생성
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)  # Create intermediate directories

plt.savefig(save_path)
print(f"Image saved to {save_path}")