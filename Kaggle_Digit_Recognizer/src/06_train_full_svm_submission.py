from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC


# =========================
# 1. 设置项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

train_path = project_root / "data" / "train.csv"
test_path = project_root / "data" / "test.csv"
sample_path = project_root / "data" / "sample_submission.csv"

model_dir = project_root / "models"
output_dir = project_root / "outputs"

model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 检查文件
# =========================

required_files = [
    train_path,
    test_path,
    sample_path
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"没有找到文件：{file_path}")


# =========================
# 3. 读取数据
# =========================

print("=" * 60)
print("正在读取训练集和测试集……")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print("训练集大小：", train_data.shape)
print("测试集大小：", test_data.shape)
print("提交示例大小：", sample_submission.shape)


# =========================
# 4. 分离特征和标签
# =========================

y_train = train_data["label"].to_numpy()

X_train = train_data.drop(
    columns=["label"]
).to_numpy(dtype=np.float32)

X_test = test_data.to_numpy(dtype=np.float32)

print("\n训练特征大小：", X_train.shape)
print("训练标签大小：", y_train.shape)
print("测试特征大小：", X_test.shape)


# =========================
# 5. 像素归一化
# =========================

X_train /= 255.0
X_test /= 255.0

print("\n训练集像素范围：", X_train.min(), "～", X_train.max())
print("测试集像素范围：", X_test.min(), "～", X_test.max())

# 删除不再使用的DataFrame，减少内存占用
del train_data
del test_data


# =========================
# 6. 创建SVM模型
# =========================

model = SVC(
    C=10.0,
    kernel="rbf",
    gamma="scale",
    cache_size=1024
)


# =========================
# 7. 使用全部42000张图片训练
# =========================

print("\n开始使用全部42000张训练图片训练SVM……")
print("这次数据比验证阶段更多，运行时间更长属于正常情况。")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\n训练完成，耗时：{training_time:.2f}秒")
print(f"训练完成，耗时：{training_time / 60:.2f}分钟")


# =========================
# 8. 预测测试集
# =========================

print("\n正在预测28000张测试图片……")

prediction_start = time.time()

test_predictions = model.predict(X_test)

prediction_time = time.time() - prediction_start

print(f"预测完成，耗时：{prediction_time:.2f}秒")
print("预测结果数量：", len(test_predictions))
print("前20个预测结果：", test_predictions[:20])


# =========================
# 9. 生成Kaggle提交文件
# =========================

submission = sample_submission.copy()

submission["Label"] = test_predictions.astype(int)


# =========================
# 10. 严格检查提交格式
# =========================

expected_columns = ["ImageId", "Label"]

if submission.columns.tolist() != expected_columns:
    raise ValueError(
        f"提交文件列名错误，当前列名：{submission.columns.tolist()}"
    )

if len(submission) != 28000:
    raise ValueError(
        f"提交文件行数错误，当前行数：{len(submission)}"
    )

if not submission["Label"].between(0, 9).all():
    raise ValueError("预测结果中存在0～9以外的数字")

if submission.isnull().sum().sum() != 0:
    raise ValueError("提交文件中存在缺失值")

print("\n提交文件检查通过")
print("提交文件大小：", submission.shape)
print("提交文件列名：", submission.columns.tolist())

print("\n提交文件前10行：")
print(submission.head(10))

print("\n预测标签分布：")
print(submission["Label"].value_counts().sort_index())


# =========================
# 11. 保存提交文件
# =========================

submission_path = (
    output_dir
    / "submission_svm_rbf.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print("\n提交文件已经保存到：")
print(submission_path)


# =========================
# 12. 保存完整训练模型
# =========================

model_path = (
    model_dir
    / "svm_rbf_full.joblib"
)

joblib.dump(model, model_path)

print("\n完整SVM模型已经保存到：")
print(model_path)

print("\n全部任务完成")
print("=" * 60)