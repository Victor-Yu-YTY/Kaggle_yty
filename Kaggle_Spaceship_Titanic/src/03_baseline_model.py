from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. 路径设置
# ============================================================

# 当前文件：
# SpaceshipTitanic/src/03_baseline_model.py
#
# parents[0]：src
# parents[1]：SpaceshipTitanic
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 数据文件路径
# ============================================================

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sample_submission_path = DATA_DIR / "sample_submission.csv"


# 检查文件是否存在
for path in [
    train_path,
    test_path,
    sample_submission_path,
]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到数据文件：{path}")


# ============================================================
# 3. 读取数据
# ============================================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(sample_submission_path)

print("=" * 80)
print("原始数据规模")
print("=" * 80)

print("训练集形状：", train_df.shape)
print("测试集形状：", test_df.shape)
print("提交示例形状：", sample_submission_df.shape)


# ============================================================
# 4. 计算完整数据中的旅行小组人数
# ============================================================

# PassengerId 格式：
# gggg_pp
#
# 为避免同一旅行小组的成员分别出现在训练集和测试集中，
# 这里将训练集和测试集的 PassengerId 合并后统一计算 GroupSize。

all_passenger_ids = pd.concat(
    [
        train_df["PassengerId"],
        test_df["PassengerId"],
    ],
    ignore_index=True,
)

all_group_ids = (
    all_passenger_ids
    .str.split("_")
    .str[0]
)

group_size_map = all_group_ids.value_counts().to_dict()


# ============================================================
# 5. 特征工程函数
# ============================================================

def create_features(
    df: pd.DataFrame,
    group_sizes: dict,
) -> pd.DataFrame:
    """
    对 Spaceship Titanic 原始数据进行基础特征工程。

    参数
    ----------
    df:
        原始乘客数据。

    group_sizes:
        旅行小组编号与小组人数之间的映射。

    返回
    -------
    pd.DataFrame:
        完成特征工程后的数据。
    """

    result = df.copy()

    # --------------------------------------------------------
    # 5.1 处理 PassengerId
    # --------------------------------------------------------

    passenger_parts = result["PassengerId"].str.split(
        "_",
        expand=True,
    )

    # 旅行小组编号
    result["GroupId"] = passenger_parts[0]

    # 乘客在小组中的序号
    result["GroupMemberNumber"] = pd.to_numeric(
        passenger_parts[1],
        errors="coerce",
    )

    # 旅行小组总人数
    result["GroupSize"] = (
        result["GroupId"]
        .map(group_sizes)
    )

    # 是否独自旅行
    result["IsAlone"] = result["GroupSize"] == 1

    # --------------------------------------------------------
    # 5.2 拆分 Cabin
    # Cabin 格式：Deck/Number/Side
    # --------------------------------------------------------

    cabin_parts = result["Cabin"].str.split(
        "/",
        expand=True,
    )

    result["CabinDeck"] = cabin_parts[0]

    result["CabinNumber"] = pd.to_numeric(
        cabin_parts[1],
        errors="coerce",
    )

    result["CabinSide"] = cabin_parts[2]

    # --------------------------------------------------------
    # 5.3 构造消费特征
    # --------------------------------------------------------

    spending_columns = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    # 总消费
    result["TotalSpend"] = (
        result[spending_columns]
        .fillna(0)
        .sum(axis=1)
    )

    # 是否完全没有消费
    result["NoSpend"] = result["TotalSpend"] == 0

    # 使用过多少类收费服务
    result["ServicesUsed"] = (
        result[spending_columns]
        .fillna(0)
        .gt(0)
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # 5.4 删除暂时不直接输入模型的字段
    # --------------------------------------------------------

    result = result.drop(
        columns=[
            "PassengerId",
            "GroupId",
            "Cabin",
            "Name",
        ],
        errors="ignore",
    )

    return result


# ============================================================
# 6. 准备训练特征和目标变量
# ============================================================

X_raw = train_df.drop(columns=["Transported"])
y = train_df["Transported"].astype(bool)

X_test_raw = test_df.copy()

X = create_features(
    X_raw,
    group_size_map,
)

X_test = create_features(
    X_test_raw,
    group_size_map,
)


print("\n" + "=" * 80)
print("特征工程结果")
print("=" * 80)

print("特征工程后的训练集形状：", X.shape)
print("特征工程后的测试集形状：", X_test.shape)

print("\n模型使用的字段：")

for index, column in enumerate(
    X.columns,
    start=1,
):
    print(f"{index:2d}. {column}")


# ============================================================
# 7. 区分数值特征和类别特征
# ============================================================

numeric_columns = X.select_dtypes(
    include=["number"],
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=[
        "object",
        "bool",
        "string",
        "category",
    ],
).columns.tolist()


print("\n" + "=" * 80)
print("字段类型")
print("=" * 80)

print("\n数值特征：")
print(numeric_columns)

print("\n类别特征：")
print(categorical_columns)


# ============================================================
# 8. 统一类别字段的数据类型
# ============================================================

# CryoSleep、VIP 等字段中同时存在：
#
# True、False、缺失值
#
# 若缺失值填充为字符串 Unknown，就会出现：
#
# bool 与 str 混合
#
# OneHotEncoder 无法处理混合类型，所以这里统一转为字符串。

for column in categorical_columns:
    X[column] = (
        X[column]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
    )

    X_test[column] = (
        X_test[column]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
    )


print("\n类别字段类型已经统一为字符串。")

print("\n类别字段的数据类型：")
print(X[categorical_columns].dtypes)


# ============================================================
# 9. 检查训练集和测试集字段
# ============================================================

if X.columns.tolist() != X_test.columns.tolist():
    raise ValueError(
        "特征工程后，训练集和测试集字段不一致。"
    )

print("\n训练集和测试集字段完全一致：True")


# ============================================================
# 10. 数值特征处理流程
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        # 数值型缺失值使用中位数填补
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),

        # 数值标准化
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# ============================================================
# 11. 类别特征处理流程
# ============================================================

# 类别字段已在前面：
#
# 1. 填充为 Unknown
# 2. 统一转换成字符串
#
# 因此这里不再使用 SimpleImputer。

categorical_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)


# ============================================================
# 12. 合并数值和类别预处理流程
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ],
    remainder="drop",
)


# ============================================================
# 13. 创建逻辑回归模型
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                solver="liblinear",
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# 14. 划分训练集和验证集
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\n" + "=" * 80)
print("训练集和验证集划分")
print("=" * 80)

print("训练子集形状：", X_train.shape)
print("验证集形状：", X_valid.shape)

print("\n训练子集目标分布：")
print(y_train.value_counts())

print("\n验证集目标分布：")
print(y_valid.value_counts())


# ============================================================
# 15. 训练逻辑回归模型
# ============================================================

print("\n开始训练逻辑回归模型……")

model.fit(
    X_train,
    y_train,
)

print("逻辑回归模型训练完成。")


# ============================================================
# 16. 在验证集上进行预测
# ============================================================

valid_predictions = model.predict(X_valid)


# ============================================================
# 17. 计算验证结果
# ============================================================

valid_accuracy = accuracy_score(
    y_valid,
    valid_predictions,
)

valid_confusion_matrix = confusion_matrix(
    y_valid,
    valid_predictions,
)

valid_classification_report = classification_report(
    y_valid,
    valid_predictions,
    digits=4,
    zero_division=0,
)


print("\n" + "=" * 80)
print("本地验证结果")
print("=" * 80)

print(f"验证集准确率：{valid_accuracy:.5f}")

print("\n混淆矩阵：")
print(valid_confusion_matrix)

print("\n混淆矩阵含义：")
print("[[实际 False 且预测 False, 实际 False 但预测 True],")
print(" [实际 True 但预测 False,  实际 True 且预测 True]]")

print("\n分类报告：")
print(valid_classification_report)


# ============================================================
# 18. 使用全部训练数据重新训练模型
# ============================================================

print("\n使用全部训练数据重新训练模型……")

model.fit(
    X,
    y,
)

print("完整模型训练完成。")


# ============================================================
# 19. 预测测试集
# ============================================================

test_predictions = model.predict(X_test)
test_predictions = test_predictions.astype(bool)


print("\n" + "=" * 80)
print("测试集预测结果")
print("=" * 80)

prediction_distribution = (
    pd.Series(
        test_predictions,
        name="Transported",
    )
    .value_counts()
)

print("预测结果分布：")
print(prediction_distribution)


# ============================================================
# 20. 生成 Kaggle 提交文件
# ============================================================

submission_df = sample_submission_df.copy()

submission_df["Transported"] = test_predictions


# 检查提交文件
if submission_df.shape != sample_submission_df.shape:
    raise ValueError(
        "生成的提交文件形状与 sample_submission.csv 不一致。"
    )

if submission_df["PassengerId"].tolist() != test_df["PassengerId"].tolist():
    raise ValueError(
        "提交文件的 PassengerId 顺序与 test.csv 不一致。"
    )


submission_output_path = (
    OUTPUT_DIR
    / "submission_logistic_baseline.csv"
)

submission_df.to_csv(
    submission_output_path,
    index=False,
)


print("\n提交文件前五行：")
print(submission_df.head())

print("\n提交文件形状：", submission_df.shape)

print(
    "提交文件已保存：",
    submission_output_path,
)


# ============================================================
# 21. 保存训练好的模型
# ============================================================

model_output_path = (
    MODEL_DIR
    / "logistic_baseline.joblib"
)

joblib.dump(
    model,
    model_output_path,
)

print(
    "模型已保存：",
    model_output_path,
)


# ============================================================
# 22. 保存本地验证指标
# ============================================================

metrics_output_path = (
    OUTPUT_DIR
    / "baseline_metrics.txt"
)

with open(
    metrics_output_path,
    mode="w",
    encoding="utf-8",
) as file:
    file.write(
        "Spaceship Titanic 逻辑回归基线模型\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"验证集准确率：{valid_accuracy:.5f}\n\n"
    )

    file.write("混淆矩阵：\n")
    file.write(str(valid_confusion_matrix))

    file.write("\n\n分类报告：\n")
    file.write(valid_classification_report)

    file.write("\n测试集预测结果分布：\n")
    file.write(str(prediction_distribution))


print(
    "验证结果已保存：",
    metrics_output_path,
)


# ============================================================
# 23. 程序结束
# ============================================================

print("\n" + "=" * 80)
print("逻辑回归基线模型运行完成")
print("=" * 80)

print(f"验证集准确率：{valid_accuracy:.5f}")
print(f"提交文件：{submission_output_path}")
print(f"模型文件：{model_output_path}")
print(f"验证指标：{metrics_output_path}")