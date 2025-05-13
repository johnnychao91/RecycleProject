# Test Accuracy 比較
# 
# | 模型              | Test Accuracy                      |
# | 全凍結-50epoch------------------------------------------|
# | MobileNetV2       | 0.8361344537815126                 |
# | EfficientNetB0    | 0.8403361344537815                 |
# | EfficientNetB1    | 0.8403361344537815                 |
# | EfficientNetB7    | 0.7773109243697479                 |
# | EfficientNetV2S   | 0.7815126050420168                 |
# | EfficientNetV2M   | 0.7415966386554622                 |
# | ConvNeXtTiny      | 0.8382352941176471                 |
# | 全訓練-50epoch------------------------------------------|
# | MobileNetV2       | 0.8466386554621849                 |
# | EfficientNetB0    | 0.8634453781512605                 |
# | EfficientNetB1    | 0.8718487394957983                 |
# | EfficientNetB7    | 0.7731092436974790                 |
# | EfficientNetV2S   | 0.8298319327731093                 |
# | EfficientNetV2M   | 0.8172268907563025                 |
# | ConvNeXtTiny      | 0.7815126050420168                 |
# | 全訓練-200epoch-----------------------------------------|
# | MobileNetV2       | 0.8298319327731093                 |
# | EfficientNetB0    |                  |
# | EfficientNetB1    | 0.8739495798319328                 |
# | EfficientNetB7    | 0.8298319327731093                 |
# | EfficientNetV2S   |                  |
# | EfficientNetV2M   |                  |
# | ConvNeXtTiny      |                  |
# | 半凍結半訓練-200epoch------------------------------------|
# | MobileNetV2       |                  |
# | EfficientNetB0    |                  |
# | EfficientNetB1    |                  |
# | EfficientNetB7    |                  |
# | EfficientNetV2S   |                  |
# | EfficientNetV2M   |                  |
# | ConvNeXtTiny      |                  |


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from pathlib import Path
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import random
from torch.backends import cudnn

# 檢查 GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# 固定隨機種子
SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

# 如果要 DataLoader worker 也固定種子
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 建議在你的 DataLoader 中這樣使用：
g = torch.Generator()
g.manual_seed(SEED)

INPUT_PATH = "./data/realwaste-main/RealWaste"
print(os.listdir(INPUT_PATH))

# 資料載入
all_classes = ['Glass', 'Metal', 'Food Organics', 'Miscellaneous Trash', 'Plastic',
               'Paper', 'Textile Trash', 'Cardboard', 'Vegetation']

def load_data():
    data = []
    for idx, cls in enumerate(all_classes):
        images = Path(f"{INPUT_PATH}/{cls}").glob("*.jpg")
        data.extend([(img, idx) for img in images])
    return pd.DataFrame(data, columns=['image', 'label'])

total_data = load_data()
train_val_df, test_df = train_test_split(total_data, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=3/17, random_state=42)

label_counts = test_df['label'].value_counts()
assert all(label_counts >= 40), "有類別樣本數不足 40！"

test_df = test_df.groupby('label').sample(n=40, random_state=42).reset_index(drop=True)

train_df = train_df.sample(frac=1., random_state=100).reset_index(drop=True)

# Dataset 定義
class WasteDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = str(self.data.iloc[idx]['image'])
        label = self.data.iloc[idx]['label']
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = WasteDataset(train_df, transform=transform)
val_dataset = WasteDataset(val_df, transform=transform)
test_dataset = WasteDataset(test_df, transform=transform)

x_train = DataLoader(train_dataset, batch_size=16, shuffle=True, worker_init_fn=seed_worker, generator=g)
x_val = DataLoader(val_dataset, batch_size=16, shuffle=False, worker_init_fn=seed_worker, generator=g)
x_test = DataLoader(test_dataset, batch_size=16, shuffle=False, worker_init_fn=seed_worker, generator=g)

model_dir = f"./models"


model = models.mobilenet_v2(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 9)
model = model.to(device)
model_dir = model_dir + "/MobileNetV2"


"""
model = models.efficientnet_b0(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 9)
model = model.to(device)
model_dir = model_dir + "/EfficientNetB0"
"""

"""
model = models.efficientnet_b1(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 9)
model = model.to(device)
model_dir = model_dir + "/EfficientNetB1"
"""

"""
model = models.efficientnet_b7(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 9)
model = model.to(device)
model_dir = model_dir + "/EfficientNetB7"
"""

"""
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 9)
model = model.to(device)
model_dir = model_dir + "/EfficientNetV2S"
"""

"""
model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 9)
model = model.to(device)
model_dir = model_dir + "EfficientNetV2M"
"""

"""
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, 9)
model = model.to(device)
model_dir = model_dir + "/ConvNeXtTiny"
"""

model_dir = model_dir + "/halfFreezeHalfTrain_200epochs"

"""
#全凍結
for param in model.features.parameters():
    param.requires_grad = False
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

best_val_acc = 0.0
initial_epochs = 200
freeze_epochs = 100

train_acc_list = []
val_acc_list = []


# 訓練與驗證迴圈
for epoch in range(initial_epochs):
    if epoch == freeze_epochs:
        print(f"解凍一半層數進行 fine-tuning at epoch {epoch}")

        total_layers = list(model.named_parameters())
        num_to_unfreeze = len(total_layers) // 2

        for name, param in total_layers[-num_to_unfreeze:]:
            param.requires_grad = True

        # 重新建立 optimizer（必須）
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(x_train):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in x_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # 根據模型名稱建立資料夾並儲存
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.pth")
        torch.save(model.state_dict(), model_path)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    
    #scheduler.step()
    
    print(f"Epoch {epoch+1}/{initial_epochs} - Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


# 測試階段
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in x_test:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

test_acc = np.mean(np.array(y_true) == np.array(y_pred))
print('Test accuracy :', test_acc)

# 繪製混淆矩陣
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=all_classes)
disp.plot(xticks_rotation=45)
plt.title("Test Set Confusion Matrix")
plt.savefig(model_dir + "/confusion_matrix.png")

# 繪製 Train / Val Accuracy 曲線
epochs = list(range(1, initial_epochs + 1))
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc_list, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc_list, label='Validation Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(model_dir + "/accuracy_curve.png")
