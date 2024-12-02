import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class CustomTrainer():
    # 모델, 하이퍼파라미터, 손실 함수, 옵티마이저 정의
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader,
        batch_size: int=64, 
        epoch: int=50, 
        learning_rate: float=1e-4,
        criterion: str="cross-entropy",
        optimizer: str="adam",
        logging_epoch: int=1,
        eval_epoch: int=1,
        save_epoch: int=1,
        save_dir: str='outputs',
        save_best_model: bool=True,
        device: str='cpu'
    ):
        available_criteria = {
            "cross-entropy": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss()
        }
        allowed_optimizers = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad,
        }    
        if criterion not in available_criteria:
            raise Exception("잘못된 손실 함수입니다.")
        if optimizer not in allowed_optimizers:
            raise Exception("잘못된 옵티마이저입니다.")
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.criterion = available_criteria[criterion]
        self.optimizer = allowed_optimizers[optimizer](model.parameters(), learning_rate)
        self.logging_epoch = logging_epoch
        self.eval_epoch = eval_epoch
        self.save_epoch = save_epoch
        self.save_dir = save_dir
        self.save_best_model = save_best_model
        self.device = device
        
        # 로그 파일 초기화
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.log_file = os.path.join(self.save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write('')  # 기존 내용 초기화
        
        
    def fit(self):
        max_test_accuracy = 0
        for i in range(1, self.epoch + 1):
            # 훈련
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (x, y) in enumerate(tqdm(self.train_loader, desc=f"Epoch {i}/{self.epoch}")):
                x, y = x.to(self.device), y.to(self.device)
                
                o, _ = self.model(x)
                loss = self.criterion(o, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, pred = o.max(1)
                total += y.shape[0]
                correct += torch.where(pred == y, 1, 0).sum().item()

            train_accuracy = 100.0 * correct / total

            if i % self.logging_epoch == 0:
                # 로그 출력 맟 기록
                log_message = (                
                    f"Epoch #{i:2d} | "
                    f"Train Loss: {train_loss / len(self.train_loader):.4f} | "
                    f"Train Accuracy: {train_accuracy:.2f}% | "
                )
                print(log_message)
                with open(self.log_file, 'a') as f:
                    f.write(log_message + '\n')

            # 검증
            if i % self.eval_epoch == 0:
                self.model.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in self.test_loader:
                        x, y = x.to(self.device), y.to(self.device)

                        o, _ = self.model(x)
                        loss = self.criterion(o, y)

                        test_loss += loss.item()
                        _, pred = o.max(1)
                        total += y.shape[0]
                        correct += torch.where(pred == y, 1, 0).sum().item()

                test_accuracy = 100.0 * correct / total

                # 로그 출력 및 기록
                log_message = (
                    f"Test Loss: {test_loss / len(self.test_loader):.4f} | "
                    f"Test Accuracy: {test_accuracy:.2f}%"
                )
                print(log_message)
                with open(self.log_file, 'a') as f:
                    f.write(log_message + '\n')
                    
                # 테스트 성능이 가장 좋은 모델 저장
                if test_accuracy > max_test_accuracy and self.save_best_model:
                    max_test_accuracy = test_accuracy  
                    save_dir = f'./{self.save_dir}/best_model'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    # 모델 및 iter 횟수 저장
                    model_path = os.path.join(save_dir, 'model.pth')
                    info_path = os.path.join(save_dir, 'model_info.json')
                    
                    torch.save(self.model.state_dict(), model_path)
                    with open(info_path, 'w') as f:
                        json.dump({"epoch": i, "test_accuracy": test_accuracy}, f)
                    
                # 저장
                if i % self.save_epoch == 0:
                    if not os.path.exists(f'./{self.save_dir}/checkpoint-{i}'):
                        os.makedirs(f'./{self.save_dir}/checkpoint-{i}')
                    torch.save(self.model.state_dict(), f'./{self.save_dir}/checkpoint-{i}/model.pth')