import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train=pd.read_csv("titanic_train.csv")

#删除不需要的列
train.drop(columns=["PassengerId","Name","Ticket","Cabin","SibSp","Parch"],axis=1,inplace=True)         
print("\n========【阶段1：原始数据删除无用列后】========")
print(f"数据集形状: {train.shape}")
print("各列空值数量：")
print(train.isna().sum())

insame_weight = ["Pclass", "Age", "Fare"]    
same_weight = ["Sex", "Embarked"]

X = train[insame_weight+same_weight].copy()
y = train["Survived"]

# ========= 划分：训练集、验证集、测试集 =========
# 第一步：分出训练集 和 剩余部分(后面拆成val+test)
X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    X, y,
    test_size=0.3,    
    random_state=42,  
    shuffle=True      
)
# 第二步：把temp对半拆为验证集、测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    shuffle=True
)
print("\n========【阶段2：train/val/test 划分完成】========")
print(f"X_train_full shape: {X_train_full.shape}, y_train_full shape:{y_train_full.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape:{y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape:{y_test.shape}")


# ========= 缺失值填充（填充统计量全部来自训练集，防止泄露） =========
# 训练集清洗
X_train_fill = X_train_full.copy()
X_train_fill["Age"] = X_train_fill["Age"].fillna(X_train_fill["Age"].mean())
X_train_fill = X_train_fill.dropna()
y_train_fill = y_train_full.loc[X_train_fill.index]
print("\n========【阶段3：训练集填充+删除空行后】========")
print(f"X_train_fill shape: {X_train_fill.shape}, y_train_fill shape:{y_train_fill.shape}")
print("X_train_fill剩余空值：")
print(X_train_fill.isna().sum())

# 验证集清洗（均值用训练集的均值！）
X_val_fill = X_val.copy()
X_val_fill["Age"] = X_val_fill["Age"].fillna(X_train_fill["Age"].mean())
X_val_fill = X_val_fill.dropna()
y_val_fill = y_val.loc[X_val_fill.index]
print("\n========【阶段4：验证集填充+删除空行后】========")
print(f"X_val_fill shape: {X_val_fill.shape}, y_val_fill shape:{y_val_fill.shape}")
print("X_val_fill剩余空值：")
print(X_val_fill.isna().sum())

# 测试集清洗
X_test_fill = X_test.copy()
X_test_fill["Age"] = X_test_fill["Age"].fillna(X_train_fill["Age"].mean())
X_test_fill = X_test_fill.dropna()
y_test_fill = y_test.loc[X_test_fill.index]
print("\n========【阶段5：测试集填充+删除空行后】========")
print(f"X_test_fill shape: {X_test_fill.shape}, y_test_fill shape:{y_test_fill.shape}")
print("X_test_fill剩余空值：")
print(X_test_fill.isna().sum())


# ========= 独热编码预处理器：只在训练集fit =========
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), same_weight)
    ],
    remainder="passthrough"  
)
X_train_1 = preprocessor.fit_transform(X_train_fill)
X_val_1 = preprocessor.transform(X_val_fill)
X_test_1 = preprocessor.transform(X_test_fill)

print("\n========【阶段6：独热编码预处理完成】========")
print(f"X_train_1(编码后训练特征) shape: {X_train_1.shape}")
print(f"X_val_1(编码后验证特征) shape: {X_val_1.shape}")
print(f"X_test_1(编码后测试特征) shape: {X_test_1.shape}")


# ======================
# 第一部分：SVM支持向量机，用验证集调参
# ======================
print("\n========【SVM 在验证集上挑选最优C】========")
C_candidate = [0.01, 0.1, 1, 10, 100]
best_val_acc_svm = 0
best_C = None

for c_val in C_candidate:
    svm_cv = SVC(C=c_val, kernel="linear", random_state=0, max_iter=1000)
    svm_cv.fit(X_train_1, y_train_fill)
    y_val_pred_svm = svm_cv.predict(X_val_1)
    val_acc = accuracy_score(y_val_fill, y_val_pred_svm)
    print(f"C={c_val} | 验证集准确率：{val_acc:.4f}")
    if val_acc > best_val_acc_svm:
        best_val_acc_svm = val_acc
        best_C = c_val

print(f"\nSVM最优超参：best_C = {best_C}，最优验证集准确率：{best_val_acc_svm:.4f}")
# 使用最优C训练最终SVM模型
svm_model = SVC(C=best_C, kernel="linear", random_state=0, max_iter=1000)
svm_model.fit(X_train_1, y_train_fill)

# 最终评估：测试集（只使用一次！）
y_test_pred_svm = svm_model.predict(X_test_1)
svm_test_acc = accuracy_score(y_test_fill, y_test_pred_svm)
print(f"\n========【SVM最终测试集评估】========")
print(f"SVM 测试集准确率：{svm_test_acc:.2f}")
print("支持向量数量：", svm_model.n_support_)


# ======================
# 第二部分：随机森林，同样用验证集调参
# ======================
print("\n========【随机森林 在验证集上挑选最优n_estimators】========")
n_est_candidate = [5,10,15,20]
best_val_acc_rf = 0
best_n_est = None

for n_est in n_est_candidate:
    rf_cv = RandomForestClassifier(n_estimators=n_est, max_depth=1, random_state=0)
    rf_cv.fit(X_train_1, y_train_fill)
    y_val_pred_rf = rf_cv.predict(X_val_1)
    val_acc = accuracy_score(y_val_fill, y_val_pred_rf)
    print(f"n_estimators={n_est} | 验证集准确率：{val_acc:.4f}")
    if val_acc > best_val_acc_rf:
        best_val_acc_rf = val_acc
        best_n_est = n_est

print(f"\n随机森林最优超参：best_n_est = {best_n_est}，最优验证集准确率：{best_val_acc_rf:.4f}")
# 用最优参数训练最终随机森林
rf_model = RandomForestClassifier(n_estimators=best_n_est, max_depth=1, random_state=0)
rf_model.fit(X_train_1, y_train_fill)

# 最终测试集评估
y_test_pred_rf = rf_model.predict(X_test_1)
rf_test_acc = accuracy_score(y_test_fill, y_test_pred_rf)
print(f"\n========【随机森林最终测试集评估】========")
print(f"随机森林 测试集准确率：{rf_test_acc:.2f}")
