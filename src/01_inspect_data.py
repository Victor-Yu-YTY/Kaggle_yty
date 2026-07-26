from pathlib import Path

import pandas as pd


# 显示所有列，避免输出时省略中间字段
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# 当前文件位于：
# SpaceshipTitanic/src/01_inspect_data.py
#
# parents[0] 是 src
# parents[1] 是 SpaceshipTitanic
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


# 三个数据文件的路径
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
submission_path = DATA_DIR / "sample_submission.csv"


# 检查文件是否存在
for path in [train_path, test_path, submission_path]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到文件：{path}")


# 读取数据
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(submission_path)


# 1. 查看数据规模
print("=" * 70)
print("1. 数据规模")
print("=" * 70)

print("训练集形状：", train_df.shape)
print("测试集形状：", test_df.shape)
print("提交示例形状：", sample_submission_df.shape)


# 2. 查看训练集前五行
print("\n" + "=" * 70)
print("2. 训练集前五行")
print("=" * 70)

print(train_df.head())


# 3. 查看列名
print("\n" + "=" * 70)
print("3. 训练集列名")
print("=" * 70)

print(train_df.columns.tolist())


# 4. 查看数据类型
print("\n" + "=" * 70)
print("4. 数据类型和非空数量")
print("=" * 70)

train_df.info()


# 5. 查看缺失值数量
print("\n" + "=" * 70)
print("5. 缺失值数量")
print("=" * 70)

missing_summary = pd.DataFrame(
    {
        "缺失数量": train_df.isna().sum(),
        "缺失比例": (train_df.isna().mean() * 100).round(2),
    }
).sort_values("缺失数量", ascending=False)

print(missing_summary)


# 6. 查看预测目标分布
print("\n" + "=" * 70)
print("6. Transported 分布")
print("=" * 70)

target_summary = pd.DataFrame(
    {
        "人数": train_df["Transported"].value_counts(),
        "比例": (
            train_df["Transported"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        ),
    }
)

print(target_summary)


# 7. 检查训练集和测试集字段是否一致
print("\n" + "=" * 70)
print("7. 训练集与测试集字段检查")
print("=" * 70)

train_features = train_df.drop(columns="Transported").columns.tolist()
test_features = test_df.columns.tolist()

print("特征字段是否一致：", train_features == test_features)


# 8. 查看提交文件格式
print("\n" + "=" * 70)
print("8. 提交文件前五行")
print("=" * 70)

print(sample_submission_df.head())