from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sample_submission_path = DATA_DIR / "sample_submission.csv"


for path in [
    train_path,
    test_path,
    sample_submission_path,
]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到数据文件：{path}")


# ============================================================
# 2. 读取原始数据
# ============================================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(sample_submission_path)


print("=" * 80)
print("原始数据规模")
print("=" * 80)

print("训练集：", train_df.shape)
print("测试集：", test_df.shape)
print("提交示例：", sample_submission_df.shape)


# ============================================================
# 3. 计算训练集和测试集共享的统计特征
# ============================================================

# 这里只使用特征信息，不使用 test.csv 的标签。
all_feature_df = pd.concat(
    [
        train_df.drop(columns=["Transported"]),
        test_df,
    ],
    ignore_index=True,
)


# ------------------------------------------------------------
# 3.1 旅行小组人数
# ------------------------------------------------------------

all_group_ids = (
    all_feature_df["PassengerId"]
    .str.split("_")
    .str[0]
)

group_size_map = (
    all_group_ids
    .value_counts()
    .to_dict()
)


# ------------------------------------------------------------
# 3.2 姓氏人数
# ------------------------------------------------------------

all_surnames = (
    all_feature_df["Name"]
    .fillna("Unknown")
    .astype(str)
    .str.split()
    .str[-1]
    .fillna("Unknown")
)

surname_size_map = (
    all_surnames
    .value_counts()
    .to_dict()
)


# ============================================================
# 4. 特征工程函数
# ============================================================

def create_features(
    df: pd.DataFrame,
    group_sizes: dict,
    surname_sizes: dict,
) -> pd.DataFrame:
    """
    构造 CatBoost 模型使用的特征。
    """

    result = df.copy()

    # --------------------------------------------------------
    # 4.1 PassengerId：旅行小组特征
    # --------------------------------------------------------

    passenger_parts = result["PassengerId"].str.split(
        "_",
        expand=True,
    )

    result["GroupId"] = passenger_parts[0]

    result["GroupMemberNumber"] = pd.to_numeric(
        passenger_parts[1],
        errors="coerce",
    )

    result["GroupSize"] = (
        result["GroupId"]
        .map(group_sizes)
    )

    result["IsAlone"] = (
        result["GroupSize"] == 1
    )

    result["GroupSizeCategory"] = pd.cut(
        result["GroupSize"],
        bins=[
            0,
            1,
            2,
            4,
            6,
            float("inf"),
        ],
        labels=[
            "Alone",
            "Two",
            "Small",
            "Medium",
            "Large",
        ],
    )


    # --------------------------------------------------------
    # 4.2 Name：姓氏和家庭特征
    # --------------------------------------------------------

    result["Surname"] = (
        result["Name"]
        .fillna("Unknown")
        .astype(str)
        .str.split()
        .str[-1]
        .fillna("Unknown")
    )

    result["SurnameSize"] = (
        result["Surname"]
        .map(surname_sizes)
        .fillna(1)
    )

    result["IsFamily"] = (
        result["SurnameSize"] > 1
    )


    # --------------------------------------------------------
    # 4.3 Cabin：甲板、编号、左右舷和舱位区域
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

    result["CabinSection"] = pd.cut(
        result["CabinNumber"],
        bins=[
            -1,
            300,
            600,
            900,
            1200,
            1500,
            1800,
            float("inf"),
        ],
        labels=[
            "0-300",
            "301-600",
            "601-900",
            "901-1200",
            "1201-1500",
            "1501-1800",
            "1801+",
        ],
    )

    result["DeckSide"] = (
        result["CabinDeck"]
        .fillna("Unknown")
        .astype(str)
        + "_"
        + result["CabinSide"]
        .fillna("Unknown")
        .astype(str)
    )

    result["HomePlanetDeck"] = (
        result["HomePlanet"]
        .fillna("Unknown")
        .astype(str)
        + "_"
        + result["CabinDeck"]
        .fillna("Unknown")
        .astype(str)
    )


    # --------------------------------------------------------
    # 4.4 年龄特征
    # --------------------------------------------------------

    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=[
            -1,
            5,
            12,
            17,
            25,
            35,
            50,
            65,
            float("inf"),
        ],
        labels=[
            "0-5",
            "6-12",
            "13-17",
            "18-25",
            "26-35",
            "36-50",
            "51-65",
            "65+",
        ],
    )

    result["IsChild"] = (
        result["Age"] < 13
    )


    # --------------------------------------------------------
    # 4.5 消费特征
    # --------------------------------------------------------

    spending_columns = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    spending_filled = (
        result[spending_columns]
        .fillna(0)
    )

    result["TotalSpend"] = (
        spending_filled
        .sum(axis=1)
    )

    result["NoSpend"] = (
        result["TotalSpend"] == 0
    )

    result["ServicesUsed"] = (
        spending_filled
        .gt(0)
        .sum(axis=1)
    )

    result["LuxurySpend"] = (
        spending_filled["Spa"]
        + spending_filled["VRDeck"]
    )

    result["BasicSpend"] = (
        spending_filled["RoomService"]
        + spending_filled["FoodCourt"]
        + spending_filled["ShoppingMall"]
    )

    result["MaximumSingleSpend"] = (
        spending_filled
        .max(axis=1)
    )

    result["AverageSpend"] = (
        spending_filled
        .mean(axis=1)
    )

    result["SpendPerUsedService"] = (
        result["TotalSpend"]
        / result["ServicesUsed"].replace(0, 1)
    )


    # --------------------------------------------------------
    # 4.6 休眠与消费的一致性特征
    # --------------------------------------------------------

    result["CryoSleepWithSpend"] = (
        (result["CryoSleep"] == True)
        & (result["TotalSpend"] > 0)
    )

    result["AwakeNoSpend"] = (
        (result["CryoSleep"] == False)
        & (result["TotalSpend"] == 0)
    )


    # --------------------------------------------------------
    # 4.7 删除已经拆分的原始字段
    # --------------------------------------------------------

    result = result.drop(
        columns=[
            "PassengerId",
            "Cabin",
            "Name",
        ],
        errors="ignore",
    )

    return result


# ============================================================
# 5. 准备训练特征、测试特征和目标变量
# ============================================================

X_raw = train_df.drop(columns=["Transported"])
X_test_raw = test_df.copy()

y = (
    train_df["Transported"]
    .astype(int)
)

# 用于分组交叉验证
groups = (
    train_df["PassengerId"]
    .str.split("_")
    .str[0]
)


X = create_features(
    X_raw,
    group_size_map,
    surname_size_map,
)

X_test = create_features(
    X_test_raw,
    group_size_map,
    surname_size_map,
)


if X.columns.tolist() != X_test.columns.tolist():
    raise ValueError(
        "特征工程后训练集和测试集字段不一致。"
    )


print("\n" + "=" * 80)
print("特征工程完成")
print("=" * 80)

print("训练特征形状：", X.shape)
print("测试特征形状：", X_test.shape)


# ============================================================
# 6. 识别类别特征
# ============================================================

categorical_columns = X.select_dtypes(
    include=[
        "object",
        "bool",
        "category",
        "string",
    ],
).columns.tolist()


# CatBoost 的类别特征不能同时包含布尔值、字符串和缺失值，
# 因此统一转换成字符串。
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


print("\n类别特征数量：", len(categorical_columns))
print("类别特征：")

for column in categorical_columns:
    print(f"- {column}")


# ============================================================
# 7. 设置分组交叉验证
# ============================================================

N_SPLITS = 5

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42,
)


# 保存每个训练样本的折外预测概率
oof_probabilities = np.zeros(
    len(X),
    dtype=float,
)

# 保存测试集的平均预测概率
test_probabilities = np.zeros(
    len(X_test),
    dtype=float,
)

fold_scores = []
fold_best_iterations = []
feature_importance_sum = np.zeros(
    X.shape[1],
    dtype=float,
)


# ============================================================
# 8. 进行 5 折交叉验证
# ============================================================

for fold, (train_index, valid_index) in enumerate(
    cv.split(X, y, groups),
    start=1,
):
    print("\n" + "=" * 80)
    print(f"开始训练第 {fold}/{N_SPLITS} 折")
    print("=" * 80)

    X_train = X.iloc[train_index].copy()
    X_valid = X.iloc[valid_index].copy()

    y_train = y.iloc[train_index]
    y_valid = y.iloc[valid_index]


    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.035,
        depth=7,

        loss_function="Logloss",
        eval_metric="Logloss",
        custom_metric=["Accuracy"],

        l2_leaf_reg=6,
        random_strength=0.6,
        bagging_temperature=0.8,

        random_seed=42 + fold,

        allow_writing_files=False,
        thread_count=-1,

        verbose=200,
    )


    model.fit(
        X_train,
        y_train,

        cat_features=categorical_columns,

        eval_set=(X_valid, y_valid),

        use_best_model=True,
        early_stopping_rounds=150,

        verbose=200,
    )


    # 验证集概率
    valid_probability = model.predict_proba(
        X_valid
    )[:, 1]

    oof_probabilities[valid_index] = (
        valid_probability
    )


    # 默认阈值 0.5 的预测结果
    valid_prediction = (
        valid_probability >= 0.5
    ).astype(int)

    fold_accuracy = accuracy_score(
        y_valid,
        valid_prediction,
    )

    fold_scores.append(fold_accuracy)

    best_iteration = model.get_best_iteration()

    fold_best_iterations.append(
        best_iteration
    )


    # 预测测试集，并进行 5 折平均
    test_probabilities += (
        model.predict_proba(X_test)[:, 1]
        / N_SPLITS
    )


    feature_importance_sum += (
        model.get_feature_importance()
        / N_SPLITS
    )


    # 保存当前折模型
    fold_model_path = (
        MODEL_DIR
        / f"catboost_fold_{fold}.cbm"
    )

    model.save_model(
        str(fold_model_path)
    )


    print(
        f"第 {fold} 折准确率："
        f"{fold_accuracy:.5f}"
    )

    print(
        f"第 {fold} 折最佳迭代次数："
        f"{best_iteration}"
    )

    print(
        f"模型已保存：{fold_model_path}"
    )


# ============================================================
# 9. 计算整体 OOF 验证结果
# ============================================================

default_predictions = (
    oof_probabilities >= 0.5
).astype(int)

default_accuracy = accuracy_score(
    y,
    default_predictions,
)


print("\n" + "=" * 80)
print("5 折交叉验证结果")
print("=" * 80)

for fold, score in enumerate(
    fold_scores,
    start=1,
):
    print(
        f"第 {fold} 折准确率：{score:.5f}"
    )

print(
    f"\n各折平均准确率："
    f"{np.mean(fold_scores):.5f}"
)

print(
    f"各折准确率标准差："
    f"{np.std(fold_scores):.5f}"
)

print(
    f"OOF 默认阈值准确率："
    f"{default_accuracy:.5f}"
)


# ============================================================
# 10. 搜索最佳概率阈值
# ============================================================

thresholds = np.arange(
    0.45,
    0.551,
    0.001,
)

threshold_scores = []

for threshold in thresholds:
    threshold_prediction = (
        oof_probabilities >= threshold
    ).astype(int)

    threshold_accuracy = accuracy_score(
        y,
        threshold_prediction,
    )

    threshold_scores.append(
        threshold_accuracy
    )


best_threshold_index = int(
    np.argmax(threshold_scores)
)

best_threshold = float(
    thresholds[best_threshold_index]
)

best_oof_accuracy = float(
    threshold_scores[best_threshold_index]
)


print("\n" + "=" * 80)
print("最佳分类阈值")
print("=" * 80)

print(
    f"默认阈值：0.500"
)

print(
    f"最佳阈值：{best_threshold:.3f}"
)

print(
    f"最佳阈值 OOF 准确率："
    f"{best_oof_accuracy:.5f}"
)


# ============================================================
# 11. 输出最终 OOF 分类结果
# ============================================================

final_oof_predictions = (
    oof_probabilities >= best_threshold
).astype(int)

final_confusion_matrix = confusion_matrix(
    y,
    final_oof_predictions,
)

final_classification_report = (
    classification_report(
        y,
        final_oof_predictions,
        digits=4,
        zero_division=0,
    )
)


print("\n混淆矩阵：")
print(final_confusion_matrix)

print("\n分类报告：")
print(final_classification_report)


# ============================================================
# 12. 生成测试集最终预测
# ============================================================

test_predictions = (
    test_probabilities >= best_threshold
).astype(bool)


print("\n测试集预测分布：")
print(
    pd.Series(
        test_predictions,
        name="Transported",
    ).value_counts()
)


# ============================================================
# 13. 生成 Kaggle 提交文件
# ============================================================

submission_df = sample_submission_df.copy()

submission_df["Transported"] = (
    test_predictions
)


if submission_df.shape != sample_submission_df.shape:
    raise ValueError(
        "提交文件形状不正确。"
    )

if (
    submission_df["PassengerId"].tolist()
    != test_df["PassengerId"].tolist()
):
    raise ValueError(
        "PassengerId 顺序不正确。"
    )


submission_path = (
    OUTPUT_DIR
    / "submission_catboost_cv.csv"
)

submission_df.to_csv(
    submission_path,
    index=False,
)


print("\n提交文件前五行：")
print(submission_df.head())

print("\n提交文件已保存：")
print(submission_path)


# ============================================================
# 14. 保存特征重要性
# ============================================================

feature_importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": feature_importance_sum,
    }
).sort_values(
    by="Importance",
    ascending=False,
)


feature_importance_path = (
    OUTPUT_DIR
    / "catboost_feature_importance.csv"
)

feature_importance_df.to_csv(
    feature_importance_path,
    index=False,
)


print("\n前 20 个重要特征：")
print(
    feature_importance_df
    .head(20)
    .to_string(index=False)
)

print("\n特征重要性已保存：")
print(feature_importance_path)


# ============================================================
# 15. 保存 OOF 预测结果
# ============================================================

oof_result_df = pd.DataFrame(
    {
        "PassengerId": train_df["PassengerId"],
        "Actual": y.astype(bool),
        "Probability": oof_probabilities,
        "Prediction": final_oof_predictions.astype(bool),
    }
)


oof_output_path = (
    OUTPUT_DIR
    / "catboost_oof_predictions.csv"
)

oof_result_df.to_csv(
    oof_output_path,
    index=False,
)


# ============================================================
# 16. 保存模型指标
# ============================================================

metrics_path = (
    OUTPUT_DIR
    / "catboost_cv_metrics.txt"
)

with open(
    metrics_path,
    mode="w",
    encoding="utf-8",
) as file:
    file.write(
        "Spaceship Titanic CatBoost 5 折模型\n"
    )

    file.write("=" * 60 + "\n\n")

    for fold, score in enumerate(
        fold_scores,
        start=1,
    ):
        file.write(
            f"第 {fold} 折准确率："
            f"{score:.5f}\n"
        )

    file.write(
        f"\n各折平均准确率："
        f"{np.mean(fold_scores):.5f}\n"
    )

    file.write(
        f"各折准确率标准差："
        f"{np.std(fold_scores):.5f}\n"
    )

    file.write(
        f"默认阈值 OOF 准确率："
        f"{default_accuracy:.5f}\n"
    )

    file.write(
        f"最佳阈值："
        f"{best_threshold:.3f}\n"
    )

    file.write(
        f"最佳阈值 OOF 准确率："
        f"{best_oof_accuracy:.5f}\n\n"
    )

    file.write("每折最佳迭代次数：\n")
    file.write(str(fold_best_iterations))

    file.write("\n\n混淆矩阵：\n")
    file.write(str(final_confusion_matrix))

    file.write("\n\n分类报告：\n")
    file.write(final_classification_report)


print("\n模型指标已保存：")
print(metrics_path)


# ============================================================
# 17. 程序结束
# ============================================================

print("\n" + "=" * 80)
print("CatBoost 交叉验证模型运行完成")
print("=" * 80)

print(
    f"平均交叉验证准确率："
    f"{np.mean(fold_scores):.5f}"
)

print(
    f"最佳阈值 OOF 准确率："
    f"{best_oof_accuracy:.5f}"
)

print(
    f"最佳阈值："
    f"{best_threshold:.3f}"
)

print(
    f"提交文件："
    f"{submission_path}"
)