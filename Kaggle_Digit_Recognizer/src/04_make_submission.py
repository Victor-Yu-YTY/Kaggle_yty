from pathlib import Path

import joblib
import pandas as pd


# =========================
# 1. 设置项目路径
# =========================

project_root = Path(__file__).resolve().parent.parent

test_path = project_root / "data" / "test.csv"
sample_path = project_root / "data" / "sample_submission.csv"
model_path = project_root / "models" / "logistic_regression.joblib"
output_dir = project_root / "outputs"

output_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 检查文件是否存在
# =========================

required_files = [test_path, sample_path, model_path]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"没有找到文件：{file_path}")


# =========================
# 3. 读取测试数据
# =========================

print("正在读取测试数据……")

test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print("测试集大小：", test_data.shape)
print("提交示例大小：", sample_submission.shape)


# =========================
# 4. 对测试数据进行归一化
# =========================

# 必须使用和训练模型时相同的处理方式
X_test = test_data.astype("float32") / 255.0

print("归一化后的最小像素值：", X_test.min().min())
print("归一化后的最大像素值：", X_test.max().max())


# =========================
# 5. 加载训练好的模型
# =========================

print("\n正在加载模型……")

model = joblib.load(model_path)

print("模型加载成功")


# =========================
# 6. 预测测试集
# =========================

print("正在预测28000张测试图片……")

test_predictions = model.predict(X_test)

print("预测完成")
print("预测结果数量：", len(test_predictions))
print("前20个预测结果：", test_predictions[:20])


# =========================
# 7. 填写提交文件
# =========================

submission = sample_submission.copy()

submission["Label"] = test_predictions


# =========================
# 8. 检查提交文件
# =========================

print("\n提交文件大小：", submission.shape)
print("提交文件前10行：")
print(submission.head(10))

print("\n预测标签分布：")
print(submission["Label"].value_counts().sort_index())


# =========================
# 9. 保存提交文件
# =========================

submission_path = output_dir / "submission_logistic_regression.csv"

submission.to_csv(
    submission_path,
    index=False
)

print(f"\n提交文件已经保存到：{submission_path}")
print("可以将该CSV文件上传到Kaggle")