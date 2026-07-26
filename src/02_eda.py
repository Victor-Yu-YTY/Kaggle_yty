# 探索性数据分析（EDA）
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. 路径和显示设置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
EDA_DIR = PROJECT_DIR / "outputs" / "eda"

EDA_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


# ============================================================
# 2. 读取训练数据
# ============================================================

train_df = pd.read_csv(DATA_DIR / "train.csv")


# ============================================================
# 3. 构造几个重要的新特征
# ============================================================

# PassengerId 格式：gggg_pp
train_df["GroupId"] = (
    train_df["PassengerId"]
    .str.split("_")
    .str[0]
)

# 每个旅行小组的人数
train_df["GroupSize"] = (
    train_df.groupby("GroupId")["PassengerId"]
    .transform("count")
)


# Cabin 格式：Deck/Number/Side
cabin_parts = train_df["Cabin"].str.split("/", expand=True)

train_df["CabinDeck"] = cabin_parts[0]
train_df["CabinNumber"] = pd.to_numeric(
    cabin_parts[1],
    errors="coerce",
)
train_df["CabinSide"] = cabin_parts[2]


# 五项消费字段
spending_columns = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

# 总消费
train_df["TotalSpend"] = (
    train_df[spending_columns]
    .fillna(0)
    .sum(axis=1)
)

# 是否完全没有消费
train_df["NoSpend"] = train_df["TotalSpend"] == 0


# ============================================================
# 4. 定义类别特征分析函数
# ============================================================

def analyse_category(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    统计不同类别的乘客数量和被传送比例。
    """
    analysis_df = df[[column, "Transported"]].copy()

    # 将缺失值单独标记出来
    analysis_df[column] = (
        analysis_df[column]
        .astype("string")
        .fillna("Missing")
    )

    result = (
        analysis_df.groupby(column)["Transported"]
        .agg(
            PassengerCount="count",
            TransportedCount="sum",
            TransportedRate="mean",
        )
        .reset_index()
    )

    result["TransportedRatePercent"] = (
        result["TransportedRate"] * 100
    ).round(2)

    return result.sort_values(
        by="TransportedRatePercent",
        ascending=False,
    )


def plot_category(
    df: pd.DataFrame,
    column: str,
    filename: str,
) -> None:
    """
    输出统计表并绘制传送比例柱状图。
    """
    result = analyse_category(df, column)

    print("\n" + "=" * 80)
    print(f"{column} 与 Transported 的关系")
    print("=" * 80)

    print(
        result[
            [
                column,
                "PassengerCount",
                "TransportedCount",
                "TransportedRatePercent",
            ]
        ].to_string(index=False)
    )

    # 保存统计表
    csv_path = EDA_DIR / f"{filename}.csv"
    result.to_csv(csv_path, index=False)

    # 绘图
    plot_data = result.set_index(column)

    plt.figure(figsize=(9, 5))

    plot_data["TransportedRatePercent"].plot(
        kind="bar"
    )

    plt.title(f"Transported Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Transported Rate (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    image_path = EDA_DIR / f"{filename}.png"

    plt.savefig(
        image_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print(f"统计表已保存：{csv_path}")
    print(f"图片已保存：{image_path}")


# ============================================================
# 5. 分析目标变量总体分布
# ============================================================

target_summary = pd.DataFrame(
    {
        "PassengerCount": train_df["Transported"].value_counts(),
        "Percentage": (
            train_df["Transported"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        ),
    }
)

print("=" * 80)
print("Transported 总体分布")
print("=" * 80)
print(target_summary)

target_summary.to_csv(
    EDA_DIR / "00_target_distribution.csv"
)


# ============================================================
# 6. 分析主要类别特征
# ============================================================

features_to_analyse = [
    ("HomePlanet", "01_home_planet"),
    ("CryoSleep", "02_cryo_sleep"),
    ("Destination", "03_destination"),
    ("VIP", "04_vip"),
    ("CabinDeck", "05_cabin_deck"),
    ("CabinSide", "06_cabin_side"),
    ("GroupSize", "07_group_size"),
    ("NoSpend", "08_no_spend"),
]

for feature, filename in features_to_analyse:
    plot_category(
        train_df,
        feature,
        filename,
    )


# ============================================================
# 7. 分析消费特征
# ============================================================

spending_summary = (
    train_df.groupby("Transported")[
        spending_columns + ["TotalSpend"]
    ]
    .median()
    .T
)

print("\n" + "=" * 80)
print("不同 Transported 状态下的消费中位数")
print("=" * 80)
print(spending_summary)

spending_summary.to_csv(
    EDA_DIR / "09_spending_median.csv"
)


# ============================================================
# 8. 分析年龄
# ============================================================

age_summary = (
    train_df.groupby("Transported")["Age"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\n" + "=" * 80)
print("不同 Transported 状态下的年龄统计")
print("=" * 80)
print(age_summary)

age_summary.to_csv(
    EDA_DIR / "10_age_summary.csv"
)


print("\nEDA 分析完成。")
print(f"分析结果保存在：{EDA_DIR}")