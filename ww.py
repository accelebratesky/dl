import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score



train=pd.read_csv("titanic_train.csv")

#删除和特征分类吧
train.drop(columns=["PassengerId","Name","Ticket","Cabin","SibSp","Parch"],axis=1,inplace=True)         

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

#该死的K折
K=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

#目标的超参数
n_trees = [2,5,7,10,12,15,17,20]
best_point = 0
best_n_tree = None

for n_tree in n_trees:
    acc_temp_list = []
    for fold_idx, (train_idx, val_idx) in enumerate(K.split(X_train, y_train)):
        fold_X_train, fold_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        fold_y_train, fold_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean"))
        ])
        cat_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, insame_weight),
                ("cat", cat_transformer, same_weight)
            ]
        )
        
        
        rf_pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("rf", RandomForestClassifier(
                n_estimators=n_tree, 
                max_depth=4,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=0
            ))
        ])
        
        rf_pipe.fit(fold_X_train, fold_y_train)
        y_fold_pred = rf_pipe.predict(fold_X_val)
        fold_acc = accuracy_score(fold_y_val, y_fold_pred)
        acc_temp_list.append(fold_acc)
    
    mean_acc = np.mean(acc_temp_list)
    print(f"n_estimators={n_tree} | 5折平均准确率：{mean_acc:.4f}")
    
    # 更新最优超参
    if mean_acc > best_mean_acc:
        best_mean_acc = mean_acc
        best_n_estimators = n_tree

print(f"\n==== K折调参完成，最优超参 ====")
print(f"最优 n_estimators = {best_n_estimators}，对应平均验证准确率：{best_mean_acc:.4f}")



num_transformer_final = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
cat_transformer_final = Pipeline(steps=[("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])
preprocessor_final = ColumnTransformer([
    ("num", num_transformer_final, insame_weight),
    ("cat", cat_transformer_final, same_weight)
])

final_rf_pipe = Pipeline(steps=[
    ("preprocess", preprocessor_final),
    ("rf", RandomForestClassifier(
        n_estimators=best_n_estimators, 
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=0
    ))
])
final_rf_pipe.fit(X_train, y_train)

# 在训练集预测
y_train_pred_rf = final_rf_pipe.predict(X_train)
rf_train_acc = accuracy_score(y_train, y_train_pred_rf)

# 在测试集预测
y_test_pred_rf = final_rf_pipe.predict(X_test)
rf_test_acc = accuracy_score(y_test, y_test_pred_rf)


print(f"随机森林训练集准确率：{rf_train_acc:.2f}")
print(f"随机森林独立测试集准确率：{rf_test_acc:.2f}")
