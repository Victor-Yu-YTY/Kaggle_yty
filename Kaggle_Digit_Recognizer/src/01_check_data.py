from pathlib import Path

import pandas as pd


# 获取项目根目录
project_root = Path(__file__).resolve().parent.parent

# 数据文件路径
train_path = project_root / "data" / "train.csv"
test_path = project_root / "data" / "test.csv"
sample_path = project_root / "data" / "sample_submission.csv"


# 检查文件是否存在
for file_path in [train_path, test_path, sample_path]:
    if not file_path.exists():
        raise FileNotFoundError(f"没有找到文件：{file_path}")


# 读取数据
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)


# 查看数据基本信息
print("=" * 50)
print("训练集大小：", train_data.shape)
print("测试集大小：", test_data.shape)
print("提交示例大小：", sample_submission.shape)

print("\n训练集前5行：")
print(train_data.head())

print("\n训练集字段：")
print(train_data.columns.tolist())

print("\n测试集前5行：")
print(test_data.head())

print("\n提交示例前5行：")
print(sample_submission.head())

print("\n训练集缺失值总数：")
print(train_data.isnull().sum().sum())

print("\n测试集缺失值总数：")
print(test_data.isnull().sum().sum())

print("\n标签分布：")
print(train_data["label"].value_counts().sort_index())

print("=" * 50)
print("数据读取和基础检查完成")