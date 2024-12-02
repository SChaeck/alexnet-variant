import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 데이터 전처리 함수
def data_load(dataset_name: str, batch_size: int):
    if dataset_name not in ['cifar-10']:
        raise Exception("잘못된 데이터셋입니다.")
    
    if dataset_name == 'cifar-10':
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))]
        )

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_dataset, train_loader, test_dataset, test_loader, classes

# 데이터 로더의 다음 배치를 가져오는 함수
def next_batch(data_loader):
    data_iter = iter(data_loader)
    one_batch = next(data_iter)
    return one_batch

# 한 쌍을 한 줄에 넣어 시각화하는 함수
def imshow_side_by_side(loader1_images, loader2_images, save_path, labels=None, classes=None, nrows=4, ncols=2, figsize=(10, 10)):
    if isinstance(loader1_images, torch.Tensor):
        loader1_images = loader1_images.permute(0, 2, 3, 1).numpy()
    if isinstance(loader2_images, torch.Tensor):
        loader2_images = loader2_images.permute(0, 2, 3, 1).numpy()

    num_images = min(len(loader1_images), len(loader2_images), nrows * ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, figsize=figsize)  # 각 쌍에 대해 2칸씩 사용

    for i in range(num_images):
        row = i // ncols
        col = (i % ncols) * 2  # 한 쌍에 두 칸 사용

        # Loader 1 이미지
        img1 = loader1_images[i]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())  # 정규화
        ax1 = axes[row, col]
        ax1.imshow(img1)
        ax1.axis('off')
        if labels is not None and classes is not None:
            ax1.set_title(f"Before: {classes[labels[i].item()]}")

        # Loader 2 이미지
        img2 = loader2_images[i]
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())  # 정규화
        ax2 = axes[row, col + 1]
        ax2.imshow(img2)
        ax2.axis('off')
        if labels is not None and classes is not None:
            ax2.set_title(f"After: {classes[labels[i].item()]}")

    # 디렉토리 생성
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Create intermediate directories

    plt.tight_layout()
    plt.savefig(save_path)  # 이미지를 파일로 저장
    print(f"Image saved to {save_path}")
    plt.close(fig)  # 메모리 누수 방지를 위해 Figure 닫기

# 전처리의 효과를 비교하기 위해 전/후 쌍을 저장하는 함수
def data_visualize_save(dataset_name: str, batch_size: int, save_path: str):
    if dataset_name not in ['cifar-10']:
        raise Exception("잘못된 데이터셋입니다.")

    if dataset_name == 'cifar-10':
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        transform_1 = transforms.Compose(
            [transforms.ToTensor()]
        )
        transform_2 = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))]
        )

        dataset_1 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_1)
        loader_1 = torch.utils.data.DataLoader(
            dataset_1,
            batch_size=8,
            shuffle=False,
        )

        dataset_2 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_2)
        loader_2 = torch.utils.data.DataLoader(
            dataset_2,
            batch_size=8,
            shuffle=False,
        )
        
        x1, y1 = next_batch(loader_1)   
        x2, y2 = next_batch(loader_2)   
        
        imshow_side_by_side(x1, x2, save_path, labels=y1, classes=classes)
