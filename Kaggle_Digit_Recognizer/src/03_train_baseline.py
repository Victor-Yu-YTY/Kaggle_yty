from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# =========================
# 1. 设置项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

train_path = project_root / "data" / "train.csv"
model_dir = project_root / "models"

# 如果 models 文件夹不存在，则自动创建
model_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取训练数据
# =========================

print("正在读取训练数据……")

train_data = pd.read_csv(train_path)

print("训练集大小：", train_data.shape)


# =========================
# 3. 分离特征和标签
# =========================

# y：每张图片的正确数字
y = train_data["label"]

# X：每张图片的784个像素
X = train_data.drop(columns=["label"])

print("特征数据大小：", X.shape)
print("标签数据大小：", y.shape)


# =========================
# 4. 像素归一化
# =========================

# 将0～255的像素值缩放到0～1
X = X.astype("float32") / 255.0

print("归一化后的最小像素值：", X.min().min())
print("归一化后的最大像素值：", X.max().max())


# =========================
# 5. 划分训练集和验证集
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("训练部分大小：", X_train.shape)
print("验证部分大小：", X_valid.shape)


# =========================
# 6. 创建逻辑回归模型
# =========================

model = LogisticRegression(
    solver="lbfgs",
    max_iter=500
)


# =========================
# 7. 训练模型
# =========================

print("\n开始训练逻辑回归模型……")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"模型训练完成，耗时：{training_time:.2f}秒")


# =========================
# 8. 验证模型
# =========================

valid_predictions = model.predict(X_valid)

accuracy = accuracy_score(y_valid, valid_predictions)

print(f"验证集准确率：{accuracy:.5f}")
print(f"验证集正确率：{accuracy * 100:.2f}%")


# =========================
# 9. 保存模型
# =========================

model_path = model_dir / "logistic_regression.joblib"

joblib.dump(model, model_path)

print(f"模型已经保存到：{model_path}")
print("\n基线模型训练完成")