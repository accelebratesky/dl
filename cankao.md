# 目录
- [Numpy](#numpy)
  - [1.1基础](#1.1基础)
  - [1.2数组运算](#12数组运算)
- [pandas](#pandas)
  - [2.1 series&dataframe](#21-seriesdataframe)
  - [2.2 数据选取](#22-数据选取)
  - [2.3pandas数据清洗](#23pandas数据清洗)
- [Matplotlib 数据可视化](#matplotlib-数据可视化)
- [其他实际应用到的操作](#其他实际应用到的操作)
  - [独热编码](#独热编码)

# Numpy
## 1.1基础
1. 导入numpy
`import numpy as np`
2. 创建ndarray数组（所有元素同一类型）
```python
#从Python列表创建数组
arr1 = np.array([1,2,3,4,5])
print("arr1 =", arr1)

#二维数组（矩阵，机器学习最常用）
arr2 = np.array([[1,2,3],
                 [4,5,6]])
print("arr2 =")
print(arr2)

# 创建全0数组
zero_arr = np.zeros((2,3)) # (行,列)
print("全0数组：")
print(zero_arr)

# 创建全1数组
one_arr = np.ones((3,2))
print("全1数组：")
print(one_arr)

# 创建连续数字数组，类似range
arange_arr = np.arange(0,10,2) # start, stop, step（范围左闭右开）
print("arange数组：", arange_arr)

```
3. 数组属性
```python
arr = np.array([[1,2,3],[4,5,6]])
print("数组形状 shape:", arr.shape)   # (2,3) 2行3列
print("维度 ndim:", arr.ndim)        # 2 二维数组
print("元素类型 dtype:", arr.dtype)  # int64 整数类型
```
4. 索引与切片

```python
#一维数组
a = np.array([10,20,30,40,50])
print(a[0])      # 取第0个元素 10
print(a[1:4])    # 切片 [20 30 40]，左闭右开
print(a[::2])    # 步长2，隔一个取一个 [10 30 50]

b = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
#二维数组
print(b[0,1])        # 第0行，第1列 → 2
print(b[1, :])       # : 代表取这一行所有列，取出第1行 [4 5 6]
print(b[:, 0])       # 取所有行，第0列（取出第0个特征！ML提取特征常用）
print(b[0:2, 1:])    # 行0~1，列1到最后

```
## 1.2数组运算
1. 逐元素四则运算
```python
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])
#`+ - * /` 都是**逐元素运算**，要求两个数组形状相同
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)   # ⚠️ 这不是矩阵乘法！是元素相乘
print("a / b =", a / b)
```
2. 广播机制（从后往前对比维度，**相等或者其中一个为 1**，就可以广播）
```python
#不同形状数组也能运算，numpy 自动把小数组扩展到大数组形状（不复制内存）
a = np.array([[1,2,3],
              [4,5,6]])
# 数组 + 标量
print(a + 10)

# 二维数组 + 一维数组（一维会广播到每一行）
b = np.array([100,200,300])
print(a + b)

结果如下
[[11 12 13]
 [14 15 16]]
[[101 202 303]
 [104 205 306]]

反例
# a.shape=(2,3)
# c.shape=(2,2)
# a + c # 直接报错！维度不匹配，无法广播
```
3. 常用统计函数(ML 看数据均值、方差)
```python
arr = np.array([[1,2,3],
                [4,5,6]])

print("全部元素求和：", arr.sum())
print("全部均值：", arr.mean())
print("最大值：", arr.max())
print("最小值：", arr.min())
print("标准差：", arr.std())

# axis 指定运算方向！超级重要
print("按行求和 axis=1：", arr.sum(axis=1)) # 每行内部求和
print("按列求和 axis=0：", arr.sum(axis=0)) # 每一列内部求和

结果如下
全部元素求和： 21
全部均值： 3.5
最大值： 6
最小值： 1
标准差： 1.707825127659933
按行求和 axis=1： [ 6 15]
按列求和 axis=0： [5 7 9]

```

4. reshape修改数组形状（不改变元素，只改视图）
```python
x = np.array([1,2,3,4,5,6])
x2 = x.reshape(2,3) # 改成2行3列
print(x2)
print(x2.shape)

# 常用技巧：-1 自动计算行数/列数
x3 = x.reshape(-1,2) # 固定2列，行数自动算
print(x3)
# reshape 总元素数量不能变！6 个元素不能 reshape 成 (2,2)!!!
结果如下
[[1 2 3]
 [4 5 6]]
(2, 3)
[[1 2]
 [3 4]
 [5 6]]

```

5. 矩阵乘法
```python
A = np.array([[1,2],
              [3,4]])
B = np.array([[5,6],
              [7,8]])

mat_mul = A @ B
mat_mul2 = np.dot(A,B)
print("矩阵乘法结果：")
print(mat_mul)
print("np.dot版本：")
print(mat_mul2)

结果如下
矩阵乘法结果：
[[19 22]
 [43 50]]
np.dot版本：
[[19 22]
 [43 50]]


```
# pandas
Pandas 用来处理表格数据（csv 数据集），机器学习拿到原始数据第一步就是用 pandas 加载查看
## 2.1 series&dataframe
1. 导入pandas
```python
import pandas as pd
```
2. Series(一维带标签的数据)

Series 是带索引的一维数组，类似带行号的一列数据

```python
# 1. 创建Series
s = pd.Series([10,20,30,40])
print("Series:\n", s)
print("s.values: ", s.values) # 取出值，变成numpy数组
print("s.index: ", s.index)   # 索引

结果
=== s ===
0    10
1    20
2    30
3    40
dtype: int64

# 2.自定义索引的Series
s2 = pd.Series([88, 92, 76, 95], index=["语文", "数学", "英语", "Python"])
print("\n=== s2 自定义索引Series ===")
print(s2)

结果
=== s2 自定义索引Series ===
语文      88
数学      92
英语      76
Python    95
dtype: int64

# 3. 取单个值（标签索引）
print("\n=== s2['数学'] ===")
print(s2["数学"])

# 4. 切片，取多个科目
print("\n=== s2[['语文','Python']] ===")
print(s2[["语文","Python"]])

# 5. Series常用属性
print("\n=== s2.index 索引 ===")
print(s2.index)

print("\n=== s2.values 值 ===")
print(s2.values)

print("\n=== s2.dtype 数据类型 ===")
print(s2.dtype)

结果
=== s2.index 索引 ===
Index(['语文', '数学', '英语', 'Python'], dtype='object')

=== s2.values 值 ===
[88 92 76 95]

=== s2.dtype 数据类型 ===
int64

```

3. DataFrame：二维表格最核心

就像 Excel 表格，有行索引、列名。机器学习数据集基本都是 DataFrame

```python
import pandas as pd


# 方式1 字典创建（推荐！机器学习构造特征表最常用）
df = pd.DataFrame({
    "身高": [170, 165, 178],
    "体重": [62, 54, 70]
}, index=["学生1", "学生2", "学生3"])

print("=== DataFrame二维表格 ===")
print(df)

结果
=== DataFrame二维表格 ===
      身高  体重
学生1  170   62
学生2  165   54
学生3  178   70

#方式2 嵌套列表创建（给 data+columns 指定列名）
data = [[170,62],[165,54],[178,70]]
df2 = pd.DataFrame(data, columns=["身高","体重"], index=["学生1","学生2","学生3"])
print("\n=== df2 嵌套列表方式 ===")
print(df2)

结果
=== df2 嵌套列表方式 ===
      身高  体重
学生1  170   62
学生2  165   54
学生3  178   70

# 查看基础属性
print("\n=== 列名 df.columns ===")
print(df.columns)

print("\n=== 行索引 df.index ===")
print(df.index)

print("\n=== 转为numpy数组 df.values ===")
print(df.values)

print("\n=== 表格形状 df.shape （行,列） ===")
print(df.shape)

结果
=== 列名 df.columns ===
Index(['身高', '体重'], dtype='object')

=== 行索引 df.index ===
Index(['学生1', '学生2', '学生3'], dtype='object')

=== 转为numpy数组 df.values ===
[[170  62]
 [165  54]
 [178  70]]

=== 表格形状 df.shape （行,列） ===
(3, 2)

```

4. 查看数据常用方法（拿到新数据集必写）

```python
print("=====基础信息=====")
print(df.head())        # 默认前5行，小数据集直接看全部
print(df.info())        # 查看类型、有没有缺失值
df.info()               # 推荐！直接调用，不要套 print，就不会出现 None
print(df.describe())    # 统计：均值、最大最小、标准差（EDA用）

结果
=== head(2) 查看前2行 ===
      身高  体重
学生1  170   62
学生2  165   54

=== info() 查看类型、缺失情况 ===
<class 'pandas.core.frame.DataFrame'>
Index: 3 entries, Index(['学生1', '学生2', '学生3'], dtype='object') #样本总数
Data columns (total 2 columns): #总共有两列特征
 #   Column  Non-Null Count  Dtype   #列名 非空数据数量 这一列的数据类型
---  ------  --------------  -----
 0   身高     3 non-null      int64
 1   体重     3 non-null      int64
dtypes: int64(2)
#`Dtype` 数据类型
   #`int64`：整数，可以直接喂给机器学习模型
   #`float64`：小数，也可以直接进模型
   #`object`：大多是字符串（文本，不能直接进模型，后面要做编码转换）
memory usage: 176.0+ bytes  #占用内存，大数据集用来评估内存压力
None #不是数据，是 print (df.info()) 这个函数的返回值

=== describe() 数值统计信息 ===
             身高         体重
count    3.000000   3.000000 #有效样本数量，如果 count 小于总行数，代表这一列有缺失
mean   171.000000  62.000000 #平均值--看特征整体水平
std      6.557439   8.000000 #标准差--代表数据波动大小
min    165.000000  54.000000
25%    167.500000  58.000000
50%    170.000000  62.000000
75%    174.000000  66.000000
max    178.000000  70.000000

```

- head ()：快速预览样本
- info ()：检查缺失值，非常重要！
- describe ()：特征统计信息

## 2.2 数据选取
后面例子基于此表

|      |height  |weight | age|
|------|--------|-------|----|
|学生1  |   172  |   65 |  19|
|学生2  |   165  |   52 |  18|
|学生3  |   178  |   71 |  20|
|学生4  |   169  |   58 |  19|
|学生5  |   180  |   75 |  18|

1. 按列选取
```python
# 1. 取出 height 这一列
col_height = df["height"]
print(col_height)
print(type(col_height)) # 查看返回对象类型

结果
学生1    172
学生2    165
学生3    178
学生4    169
学生5    180
Name: height, dtype: int64
pandas.core.series.Series

# 2. 同时取出 height 和 age 两列
cols = df[["height", "age"]] # 顺序可以随便写：`df[["age","height"]]` 会调换两列顺序
print(cols)
print(type(cols))

# 提取 多列 df [["列 1","列 2"] ]（两层方括号！外层代表选取，内层是列名列表）

运行结果

      height  age
学生1     172   19
学生2     165   18
学生3     178   20
学生4     169   19
学生5     180   18
pandas.core.frame.DataFrame
```

2. loc（按标签名字选取）
语法：`df.loc[行标签, 列标签]`
```python
# 1 取一整行
res = df.loc["学生3"]
print(res)
print(type(res))

结果
height    178
weight     71
age        20
Name: 学生3, dtype: int64
pandas.core.series.Series

# 2 指定某一行+某一列，取单个值
res = df.loc["学生3", "weight"]
print(res)

结果
71

# 3 多行+多列
res = df.loc[ ["学生1","学生5"], ["height","weight"] ]
print(res)

结果
      height  weight
学生1     172      65
学生5     180      75

```

3. iloc(按位置数字下表选取)
语法：`df.iloc[行下标, 列下标]`
```python
#示例 1：取第 4 行（学生 5），全部列
res = df.iloc[4]
print(res)

结果
height    180
weight     75
age        18
Name: 学生5, dtype: int64

#示例 2：取第 4 行，第 2 列（学生 5 的 age）
res = df.iloc[4, 2]
print(res)

结果
18

#示例 3：多行多列，取 0、1 行，0、1 列
res = df.iloc[ [0,1], [0,1] ]
print(res)

结果
      height  weight
学生1     172      65
学生2     165      52

#示例4：全部行，只取height这一列
res = df.loc[:, "height"]
print(res)

结果
学生1    172
学生2    165
学生3    178
学生4    169
学生5    180
Name: height, dtype: int64

```

4. 条件筛选
任务：筛选 weight > 60 的全部学生样本
原理：先构造布尔条件，再用条件过滤表格


```python
# 示例 1：先看布尔掩码（True/False）
cond = df["weight"] > 60
print(cond)

结果
学生1     True
学生2    False
学生3     True
学生4    False
学生5     True
Name: weight, dtype: bool

# 示例 2：把掩码传入表格，筛选数据
res = df[cond]
print(res)

结果
      height  weight  age
学生1     172      65   19
学生3     178      71   20
学生5     180      75   18

# 多条件筛选（用 & 代表并且，| 代表或者，每个条件加括号）

res = df[ (df["weight"]>60) & (df["height"]>175) ]# 体重>60 并且 身高>175
print(res)

结果
      height  weight  age
学生3     178      71   20
学生5     180      75   18
```
## 2.3pandas数据清洗
沿用前面的表格
```python
import pandas as pd
data=[[172,65,19],[165,52,18],[178,71,20],[169,58,19],[180,75,18]]
df=pd.DataFrame(data,columns=["height","weight","age"],index=["学生1","学生2","学生3","学生4","学生5"])
```

|      |height  |weight | age|
|------|--------|-------|----|
|学生1  |   172  |   65 |  19|
|学生2  |   165  |   52 |  18|
|学生3  |   178  |   71 |  20|
|学生4  |   169  |   58 |  19|
|学生5  |   180  |   75 |  18|

1. 新增一列
语法：`df["新列名"] = 值`

```python
# 示例 1：整列赋同一个值
df["gender"] = "male"
print(df)

结果
      height  weight  age gender
学生1     172      65   19   male
学生2     165      52   18   male
学生3     178      71   20   male
学生4     169      58   19   male
学生5     180      75   18   male

# 示例 2：用已有列计算生成新特征
df["bmi"] = df["weight"] / (df["height"] / 100) **2  # height单位是cm，除以100转米
print(df[["height","weight","bmi"]])

结果
      height  weight        bmi
学生1     172      65  21.971332
学生2     165      52  19.095041
学生3     178      71  22.392645
学生4     169      58  20.305505
学生5     180      75  23.148148

```
2. 删除列/行

(1) `drop()`，`axis=1` 删除列；`axis=0` 删除行(不写的话默认删行)

(2) `inplace=True` 代表**直接修改原 df**；不加的话只返回新表格，原数据不变

(**尽量少用 `inplace=True`**

机器学习里，保留原始数据是好习惯，一般写成 `新df = 原df.操作()`，避免原始数据被意外覆盖、出错后只能重跑前面代码)
```python

# 删除列示例：删掉 bmi 列
df_drop_col = df.drop("bmi", axis=1)
print(df_drop_col)

# 删除行示例：删掉学生 2 这一行
df_drop_row = df.drop("学生2", axis=0)
print(df_drop_row)

#inplace=True：直接修改原df_miss，删掉bmi列
df_miss.drop("bmi", axis=1, inplace=True)#执行完，df_miss本身就已经去掉bmi了，不需要赋值


```

3. 缺失值处理（真实数据集大量存在空值 `NaN`（Not a Number））
先手动造一个带缺失值的 df 用来演示

|      |height|  weight | age|
|------|------|---------|----|
学生1   |172.0    |65.0   |19|
学生2   |165.0    |NaN    |18|
学生3   |178.0    |71.0   |20|
学生4   |NaN      |58.0   |19|
学生5   |180.0    |75.0   |18|
```python
import numpy as np
data_miss=[[172,65,19],[165,np.nan,18],[178,71,20],[np.nan,58,19],[180,75,18]]
df_miss=pd.DataFrame(data_miss,columns=["height","weight","age"],index=["学生1","学生2","学生3","学生4","学生5"])
print(df_miss)


# 3.1 查看哪里是缺失值 `isna()`
print(df_miss.isna())

结果
       height  weight    age
学生1   False   False  False
学生2   False    True  False #ture是空值
学生3   False   False  False
学生4    True   False  False
学生5   False   False  False

# 统计每一列缺失数量
print(df_miss.isna().sum())

结果
height    1
weight    1
age       0
dtype: int64

# 3.2 直接删除含有缺失值的行 dropna()----适合缺失样本很少的场景
df_del = df_miss.dropna()
print(df_del)

结果
      height  weight  age
学生1   172.0    65.0   19
学生3   178.0    71.0   20
学生5   180.0    75.0   18

#3.3 填充缺失值 `fillna()`（机器学习更常用，不丢样本）
df_fill = df_miss.fillna(df_miss.mean())
print(df_fill)

结果
      height  weight  age
学生1   172.0    65.0   19
学生2   165.0    67.2   18
学生3   178.0    71.0   20
学生4   173.75   58.0   19
学生5   180.0    75.0   18   #分类数据一般用众数填充；数值特征常用均值 / 中位数填充

#3.4.1 单独对每列填充
df_fill = df_miss.copy()# 复制一份，不破坏原始数据
df_fill["height"] = df_fill["height"].fillna(df_fill["height"].median())# height列 → 中位数填充
df_fill["weight"] = df_fill["weight"].fillna(df_fill["weight"].mean())# weight列 → 均值填充
print(df_fill)

#3.4.2字典方式一次性fillna（`fillna(字典)`，key = 列名，value = 填充值）
df_fill = df_miss.copy()
fill_rule = {
    "height": df_fill["height"].median(),
    "weight": df_fill["weight"].mean()
}
df_fill = df_fill.fillna(fill_rule)
print(df_fill)

结果（3.4）
      height  weight  age
学生1  172.00    65.0   19
学生2  165.00    67.2   18
学生3  178.00    71.0   20
学生4  172.00    58.0   19
学生5  180.00    75.0   18

'''
提醒
1. 一定要先 `.copy()`，不然会直接改动原始 df_miss，原始数据丢失
2. 不要直接写 `df_miss.fillna()`，尽量赋值给新变量，保持原始数据不变（不推荐 inplace=True）
3. 只能填充**本列的统计量**，不能拿 A 列均值去填 B 列
'''
```
4. csv文件读写（纯文本文件）
```python
# 把df保存成csv文件
df.to_csv("student.csv", encoding="utf-8-sig")

# 读取csv文件
df_read = pd.read_csv("student.csv", index_col=0) # index_col=0 把第一列作为行索引
print(df_read)

```
# Matplotlib 数据可视化
```python
      height  weight  age        bmi
学生1  172.00    65.0   19  21.971332
学生2  165.00    67.2   18  24.661157
学生3  178.00    71.0   20  22.392645
学生4  172.00    58.0   19  19.628099
学生5  180.00    75.0   18  23.148148
```
1. 导入库
```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"] #防止画图中文乱码
plt.rcParams["axes.unicode_minus"] = False   #防止画图中文乱码
```
2. 散点图
```python
# 散点图：x=height身高，y=weight体重
plt.scatter(df_fill["height"], df_fill["weight"])
plt.xlabel("身高(cm)")    # x轴名称
plt.ylabel("体重(kg)")    # y轴名称
plt.title("身高-体重散点图") # 图标题
plt.show() # 弹出图片窗口
```

2. 直方图
```python
plt.hist(df_fill["height"], bins=3) # bins=分组数量，根据数据的最大值、最小值，平均切分成 bins 个区间
plt.xlabel("身高")
plt.ylabel("样本数量")
plt.title("身高分布直方图")
plt.show()

# 手动指定分割边界，替代bins=3
bins_edge = [160,170,175,185]
plt.hist(df_fill["height"], bins=bins_edge)

```

3. 折线图
```python
plt.plot(df_fill["age"], df_fill["bmi"], marker="o") # marker 加上圆点标记
plt.xlabel("年龄")
plt.ylabel("BMI")
plt.title("年龄-BMI折线图")
plt.show()

```

4. 子图subplot(一张画布放多张图)
```python
#写法1
plt.subplot(2,1,1)# 创建画布，2行1列，第1张图
plt.scatter(df_fill["height"],df_fill["weight"])
plt.title("身高体重散点图")

plt.subplot(2,1,2)# 2行1列，第2张图
plt.hist(df_fill["bmi"])#省略 bins 参数时：plt.hist(df_fill["height"])matplotlib 会用自动分箱算法（默认是 Sturges 公式）
plt.title("BMI分布直方图")

plt.tight_layout() # 自动调整子图间距，防止文字重叠
plt.show()

#写法2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# 创建画布fig，划分2行1列共2张子图，返回画布对象fig、子图对象ax1(上图) ax2(下图)
#figsize=(8,6) 单位英寸：图片宽8英寸，高6英寸

ax1.scatter(df_fill["height"], df_fill["bmi"])
ax1.set_xlabel("Height")
ax1.set_ylabel("BMI")
ax1.set_title("Height vs BMI")


ax2.hist(df_fill["bmi"], bins=3)
ax2.set_xlabel("BMI")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of BMI")

plt.tight_layout()
plt.show()
```



# 其他实际应用到的操作
## 独热编码
### 单个特征
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({"color":["red","blue","green","red"]})

# pandas快速独热
df_onehot = pd.get_dummies(df["color"])
print(df_onehot)

# sklearn版本（适合流水线pipeline）
enc = OneHotEncoder(sparse_output=False)
arr = enc.fit_transform(df[["color"]])
print(arr)
```
**创建一个独热编码器对象，设置参数**

- `OneHotEncoder()`：sklearn 提供的独热编码类
- `sparse_output=False`
  - 默认`sparse_output=True`：输出**稀疏矩阵**（只保存非 0 元素，省内存，适合超大维度）
  - `sparse_output=False`：**输出普通 numpy 二维数组**，打印的时候直接看到0和1，方便调试查看结果。

> 
> 小数据集、学习阶段一般写`False`；真实大数据训练可以改成`True`节省内存。

- `fit_transform` = `fit()` + `transform()` 合并写法

  - **fit(df[["color"]])**：**学习**这一列`color`里面所有类别（red/blue/green），把类别信息保存在`ohe`这个编码器里面。
  -  **transform(df[["color"]])**：使用刚刚`fit`学到的规则，把 color 这一列转换成独热 0/1 矩阵
    -  结果存入`res_arr`，是 numpy 数组

> 
>  重点：`df[["color"]]` 两层中括号，取出来是**DataFrame 二维结构**。如果写成一层`df["color"]`是一维 Series，会直接报错！sklearn 的转换器**必须输入二维数据**。
### 多个特征
```python   
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 构造样例数据：2个分类特征 color、city；1个数值特征 age
df = pd.DataFrame({
    "color": ["red", "blue", "green", "red"],
    "city": ["beijing", "shanghai", "beijing", "guangzhou"],
    "age": [22, 25, 21, 28]
})

# ========= 重点 =========
# 定义：哪些列是分类列，交给OneHot；剩下的列保持原样
cat_cols = ["color", "city"]  # 需要同时独热的多个特征
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"  # remainder="passthrough"：不在cat_cols里的列直接保留
)

# 训练集fit，train/test分开场景！
X_train = preprocessor.fit_transform(df)
print(preprocessor.get_feature_names_out()) # 查看生成后的所有列名
print(X_train)
```
**参数解释**

- `cat_cols = ["color","city"]`：直接写你所有要独热的列，**不用写多遍 OneHot**
- `remainder="passthrough"`：不在 cat_cols 的列原样保留（比如 age 数值列）；如果写`remainder="drop"`，直接丢掉其他列
- `get_feature_names_out()`：查看编码后新特征名字，解决 numpy 数组看不到列名的痛点

### 数字特征标准化&非数字特征独热编码
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.DataFrame({
    "color": ["red", "blue", "green", "red"],
    "city": ["beijing", "shanghai", "beijing", "guangzhou"],
    "age": [22, 25, 21, 28],
    "salary": [8000,12000,7000,15000]
})#这是一个例子，实际读取了csv或者exi就不用了

cat_cols = ["color", "city"]    # 多个分类特征 → OneHot
num_cols = ["age", "salary"]    # 多个数值特征 → Z-score标准化

preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ("scaler", StandardScaler(), num_cols)
    ]
)

X_result = preprocessor.fit_transform(df)#(训练集fit_transform测试集transform)
print(preprocessor.get_feature_names_out())#检查一下分类有没有问题
```
 
## 标准化和归一化
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.DataFrame({"age":[20,25,30,35,40]})

# Z-score标准化
std_scaler = StandardScaler()
data["age_std"] = std_scaler.fit_transform(data[["age"]])

# MinMax归一化
mm_scaler = MinMaxScaler()
data["age_mm"] = mm_scaler.fit_transform(data[["age"]])

print(data)
```

## 数据划分
```python
import numpy as np
from sklearn.model_selection import train_test_split
```
### 只划分为训练集和测试集（后续K折）
```python
# 模拟数据
X = np.random.rand(1000, 5)  # 1000样本，5个特征
y = np.random.randint(0,2, size=1000)

# 划分：训练80%，测试20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 测试集占总数据20%
    random_state=42,  # 随机种子，保证每次划分结果固定
    shuffle=True      # 打乱数据，避免顺序带来偏差
)

X_train_fill = X_train.copy()
X_train_fill["Age"] = X_train_fill["Age"].fillna(X_train_fill["Age"].mean())
X_train_fill = X_train_fill.dropna()

X_test_fill = X_test.copy()
X_test_fill["Age"] = X_test_fill["Age"].fillna(X_train_fill["Age"].mean())
X_test_fill = X_test_fill.dropna()

y_dev_fill = y_train.loc[X_train_fill.index]
y_test_fill = y_test.loc[X_test_fill.index]


print(f"训练集：{X_train.shape}")
print(f"测试集：{X_test.shape}")
```
### 常规划分为训练集、验证集和测试集
```python
# 模拟数据
X = np.random.rand(1000, 5)
y = np.random.randint(0,2, size=1000)

# 第一步：分出测试集（15%），剩余85%留作训练+验证
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    shuffle=True
)

# 第二步：把temp集再切分，得到训练集70%、验证集15%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.1765,   # 0.15 / 0.85 ≈0.1765
    random_state=42,
    shuffle=True
)

print(f"训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")
print(f"测试集: {X_test.shape}")
```

## K折交叉验证

```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
```
### 普通K折
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_list = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    acc_list.append(acc)

print("每折准确率：", acc_list)
print("5折平均准确率：", np.mean(acc_list))
```

### 分层K折
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list2 = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    acc_list2.append(acc)

print("分层K折每折准确率：", acc_list2)
print("分层K折平均准确率：", np.mean(acc_list2))
```
- StratifiedKFold（分层 K 折）：强制每一折里面各类别的比例，和原始数据集一模一样