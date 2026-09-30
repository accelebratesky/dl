import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ========== 1.读取原始数据 ==========
df_raw = pd.read_csv("titanic_train.csv")
df = df_raw.copy()

# ========== 2.【重点：构造新特征】 ==========
# ① 家庭总人数 = 兄弟姐妹SibSp + 父母子女Parch + 自己
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# ② 是否独自一人：家庭人数=1就是独自
df["IsAlone"] = 0
df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

# ③ 姓名提取头衔Title（Mr/Mrs/Miss，非常强力特征！）
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
# 合并稀有头衔
title_mapping = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Rev": "Other",
    "Dr": "Other",
    "Major": "Other",
    "Col": "Other",
    "Capt": "Other",
    "Lady": "Other",
    "Countess": "Other"
}
df["Title"] = df["Title"].map(title_mapping)

# ④ 票价除以家庭人数，人均票价
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# ⑤ 年龄分组（分箱，减少年龄噪声）
df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,18,30,50,100], labels=["Child","Teen","Young","Adult","Elder"])

# ========== 3.特征工程：选特征、填充缺失值、编码 ==========
# 选用原始特征 + 我们新建的特征
features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
            "FamilySize","IsAlone","Title","FarePerPerson","AgeBin"]

X = df[features].copy()
y = df["Survived"]

# 缺失值填充
X["Age"].fillna(X["Age"].median(), inplace=True)
X["FarePerPerson"].fillna(X["FarePerPerson"].median(), inplace=True)
X["Embarked"].fillna(X["Embarked"].mode()[0], inplace=True)

# 类别特征独热编码
X = pd.get_dummies(X, columns=["Sex","Embarked","Title","AgeBin"], drop_first=True)

# ========== 4.划分训练集、独立测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 5.K折交叉验证（只在训练集上做） ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=kf, scoring="accuracy")

print("="*50)
print(f"5折每折准确率：{cv_scores.round(4)}")
print(f"5折平均准确率：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("="*50)

# ========== 6.最终模型评估（在holdout测试集） ==========
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"测试集准确率：{accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 输出特征重要性，看哪些新建特征贡献大
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print("\n特征重要性排序：")
print(feature_importance)
