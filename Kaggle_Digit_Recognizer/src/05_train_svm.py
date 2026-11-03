from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# =========================
# 1. 设置项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

train_path = project_root / "data" / "train.csv"
model_dir = project_root / "models"
output_dir = project_root / "outputs"

model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 检查训练文件
# =========================

if not train_path.exists():
    raise FileNotFoundError(f"没有找到训练文件：{train_path}")


# =========================
# 3. 读取训练数据
# =========================

print("=" * 60)
print("正在读取训练数据……")

train_data = pd.read_csv(train_path)

print("训练集大小：", train_data.shape)


# =========================
# 4. 分离特征和标签
# =========================

y = train_data["label"].to_numpy()

X = train_data.drop(columns=["label"]).to_numpy(
    dtype=np.float32
)

print("特征数据大小：", X.shape)
print("标签数据大小：", y.shape)


# =========================
# 5. 像素归一化
# =========================

X /= 255.0

print("归一化后的最小像素值：", X.min())
print("归一化后的最大像素值：", X.max())


# 删除原始DataFrame，减少内存占用
del train_data


# =========================
# 6. 划分训练集和验证集
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n训练部分大小：", X_train.shape)
print("验证部分大小：", X_valid.shape)


# =========================
# 7. 创建RBF-SVM模型
# =========================

model = SVC(
    C=10.0,
    kernel="rbf",
    gamma="scale",
    cache_size=1024
)

print("\n模型参数：")
print("kernel = rbf")
print("C = 10.0")
print("gamma = scale")
print("cache_size = 1024 MB")


# =========================
# 8. 训练模型
# =========================

print("\n开始训练RBF-SVM模型……")
print("该模型比逻辑回归慢，运行数分钟属于正常现象。")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\n模型训练完成，耗时：{training_time:.2f}秒")
print(f"模型训练完成，耗时：{training_time / 60:.2f}分钟")


# =========================
# 9. 验证集预测
# =========================

print("\n正在预测验证集……")

prediction_start = time.time()

valid_predictions = model.predict(X_valid)

prediction_time = time.time() - prediction_start

print(f"验证集预测完成，耗时：{prediction_time:.2f}秒")


# =========================
# 10. 计算准确率
# =========================

accuracy = accuracy_score(
    y_valid,
    valid_predictions
)

print("\n" + "=" * 60)
print(f"SVM验证集准确率：{accuracy:.5f}")
print(f"SVM验证集正确率：{accuracy * 100:.2f}%")

# 与原逻辑回归Kaggle分数作简单对照
baseline_score = 0.91550
improvement = accuracy - baseline_score

print(f"相对0.91550基线提高：{improvement:.5f}")
print(f"提高约：{improvement * 100:.2f}个百分点")


# =========================
# 11. 输出分类报告
# =========================

print("\n各数字分类结果：")

report = classification_report(
    y_valid,
    valid_predictions,
    digits=4
)

print(report)


# =========================
# 12. 保存验证集预测结果
# =========================

validation_result = pd.DataFrame({
    "TrueLabel": y_valid,
    "PredictedLabel": valid_predictions
})

validation_result["Correct"] = (
    validation_result["TrueLabel"]
    == validation_result["PredictedLabel"]
)

validation_path = (
    output_dir
    / "svm_validation_predictions.csv"
)

validation_result.to_csv(
    validation_path,
    index=False
)

print(
    "验证结果已经保存到：",
    validation_path
)


# =========================
# 13. 保存模型
# =========================

model_path = (
    model_dir
    / "svm_rbf_validation.joblib"
)

joblib.dump(model, model_path)

print("验证模型已经保存到：", model_path)

print("\nRBF-SVM本地验证完成")
print("=" * 60)