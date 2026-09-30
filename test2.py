import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


train=pd.read_csv("titanic_train.csv")

#删除不需要的列
train.drop(columns=["PassengerId","Name","Ticket","Cabin","SibSp","Parch"],axis=1,inplace=True)         

#观察一下哪些数据需要处理吧
#print(train.isna().sum())

insame_weight = ["Pclass", "Age", "Fare"]    
same_weight = ["Sex", "Embarked"]


X = train[insame_weight+same_weight].copy()
y = train["Survived"]

#数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    
    random_state=42,  
    shuffle=True,
    stratify=y      
)
print(f"\n划分完成，训练集大小：{X_train.shape}, 测试集大小：{X_test.shape}")

#数据的填补和删减

X_train_fill = X_train.copy()
X_train_fill["Age"] = X_train_fill["Age"].fillna(X_train_fill["Age"].mean())
X_train_fill = X_train_fill.dropna()

X_test_fill = X_test.copy()
X_test_fill["Age"] = X_test_fill["Age"].fillna(X_train_fill["Age"].mean())
X_test_fill = X_test_fill.dropna()

y_train_fill = y_train.loc[X_train_fill.index]
y_test_fill = y_test.loc[X_test_fill.index]


#对特殊数据进行独热编码

preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), same_weight)
    ],
    remainder="passthrough"  
)
X_train_1= preprocessor.fit_transform(X_train_fill)
X_test_1= preprocessor.transform(X_test_fill)

#print(preprocessor.get_feature_names_out())

# 分层K折
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 待尝试的超参数候选列表
n_estimators_candidate = [2,5,7,10,12,15,18,20,21,22,25,30]
best_mean_acc = 0
best_n_estimators = None

for n_est in n_estimators_candidate:
    acc_temp_list = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_1, y_train_fill)):
        fold_X_train, fold_X_val = X_train_1[train_idx], X_train_1[val_idx]
        fold_y_train, fold_y_val = y_train_fill.iloc[train_idx], y_train_fill.iloc[val_idx]
        
        # K折内部：用当前候选超参构建随机森林
        rf_cv = RandomForestClassifier(n_estimators=n_est, max_depth=1, random_state=0)
        rf_cv.fit(fold_X_train, fold_y_train)
        y_fold_pred = rf_cv.predict(fold_X_val)
        fold_acc = accuracy_score(fold_y_val, y_fold_pred)
        acc_temp_list.append(fold_acc)
    
    mean_acc = np.mean(acc_temp_list)
    print(f"n_estimators={n_est} | 5折平均准确率：{mean_acc:.4f}")
    
    # 更新最优超参
    if mean_acc > best_mean_acc:
        best_mean_acc = mean_acc
        best_n_estimators = n_est

print(f"\n==== K折调参完成，最优超参 ====")
print(f"最优 n_estimators = {best_n_estimators}，对应平均验证准确率：{best_mean_acc:.4f}")


rf_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=1, random_state=0)
rf_model.fit(X_train_1, y_train_fill)

y_train_pred_rf = rf_model.predict(X_train_1)
rf_train_acc = accuracy_score(y_train_fill, y_train_pred_rf)

print("\n========【最终随机森林（使用K折选出的最优超参）】========")
print(f"随机森林训练集准确率：{rf_train_acc:.2f}")

# 查看每棵独立的树
for idx,tree in enumerate(rf_model.estimators_):
    print(f"\n第{idx+1}棵决策树")
    tree_pred_rf = tree.predict(X_train_1)
    print(tree_pred_rf[:10])
