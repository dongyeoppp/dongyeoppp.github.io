```
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

```train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
/kaggle/input/titanic/train.csv
/kaggle/input/titanic/test.csv
/kaggle/input/titanic/gender_submission.csv
```

```
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
```

# 데이터를 가져와 로드한다.

```
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
```

# 교육세트의 상위 몇 행을 살펴보기

```
train_data.head()
```

| PassengerId | Survived | Pclass | Name |                                               Sex |    Age | SibSp | Parch | Ticket |             Fare |   Cabin | Embarked |      |
| ----------: | -------: | -----: | ---: | ------------------------------------------------: | -----: | ----: | ----: | -----: | ---------------: | ------: | -------: | ---- |
|           0 |        1 |      0 |    3 |                           Braund, Mr. Owen Harris |   male |  22.0 |     1 |      0 |        A/5 21171 |  7.2500 |      NaN | S    |
|           1 |        2 |      1 |    1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female |  38.0 |     1 |      0 |         PC 17599 | 71.2833 |      C85 | C    |
|           2 |        3 |      1 |    3 |                            Heikkinen, Miss. Laina | female |  26.0 |     0 |      0 | STON/O2. 3101282 |  7.9250 |      NaN | S    |
|           3 |        4 |      1 |    1 |      Futrelle, Mrs. Jacques Heath (Lily May Peel) | female |  35.0 |     1 |      0 |           113803 | 53.1000 |     C123 | S    |
|           4 |        5 |      0 |    3 |                          Allen, Mr. William Henry |   male |  35.0 |     0 |      0 |           373450 |  8.0500 |      NaN | S    |

```
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
```

# 누락된 데이터가 얼마나 되는지 자세히 알아보자

```
train_data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 1 to 891
Data columns (total 12 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Survived   891 non-null    int64  
 1   Pclass     891 non-null    int64  
 2   Name       891 non-null    object 
 3   Sex        891 non-null    object 
 4   Age        714 non-null    float64
 5   SibSp      891 non-null    int64  
 6   Parch      891 non-null    int64  
 7   Ticket     891 non-null    object 
 8   Fare       891 non-null    float64
 9   Cabin      204 non-null    object 
 10  Embarked   889 non-null    object 
 11  AgeBucket  714 non-null    float64
dtypes: float64(3), int64(4), object(5)
memory usage: 90.5+ KB
```

```
train_data[train_data["Sex"]=="female"]["Age"].median()
```

```
27.0
```

# 수치 속성

```
train_data.describe()
```

|       |   Survived |     Pclass |        Age |      SibSp |      Parch |       Fare |
| ----: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| count | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
|  mean |   0.383838 |   2.308642 |  29.699118 |   0.523008 |   0.381594 |  32.204208 |
|   std |   0.486592 |   0.836071 |  14.526497 |   1.102743 |   0.806057 |  49.693429 |
|   min |   0.000000 |   1.000000 |   0.420000 |   0.000000 |   0.000000 |   0.000000 |
|   25% |   0.000000 |   2.000000 |  20.125000 |   0.000000 |   0.000000 |   7.910400 |
|   50% |   0.000000 |   3.000000 |  28.000000 |   0.000000 |   0.000000 |  14.454200 |
|   75% |   1.000000 |   3.000000 |  38.000000 |   1.000000 |   0.000000 |  31.000000 |
|   max |   1.000000 |   3.000000 |  80.000000 |   8.000000 |   6.000000 | 512.329200 |

# 목표값이 0 또는 1인지 확인

```
train_data["Survived"].value_counts()
```

```
0    549
1    342
Name: Survived, dtype: int64
```

```
train_data["Pclass"].value_counts()
```

```
3    491
1    216
2    184
Name: Pclass, dtype: int64
```

```
train_data["Sex"].value_counts()
```

```
male      577
female    314
Name: Sex, dtype: int64
```

```
train_data["Embarked"].value_counts()
```

```
S    644
C    168
Q     77
Name: Embarked, dtype: int64
```

# 전처리 파이프라인 생성

```
from sklearn.preprocessing import StandardScaler
```

# 이제 수치 속성을 위한 파이프라인부터 시작하여 전처리 파이프라인을 구축

```
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
```

# 이제 범주 속성을 위한 파이프라인을 구축

```
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),    
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
```

# 마지막으로 수치 및 범주형 파이프라인에 가입

```
from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
```

```
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
```

```
array([[-0.56573646,  0.43279337, -0.47367361, ...,  0.        ,
         0.        ,  1.        ],
       [ 0.66386103,  0.43279337, -0.47367361, ...,  1.        ,
         0.        ,  0.        ],
       [-0.25833709, -0.4745452 , -0.47367361, ...,  0.        ,
         0.        ,  1.        ],
       ...,
       [-0.1046374 ,  0.43279337,  2.00893337, ...,  0.        ,
         0.        ,  1.        ],
       [-0.25833709, -0.4745452 , -0.47367361, ...,  1.        ,
         0.        ,  0.        ],
       [ 0.20276197, -0.4745452 , -0.47367361, ...,  0.        ,
         1.        ,  0.        ]])
```

# 라벨 받기

```
y_train = train_data["Survived"]
```

```
from sklearn.ensemble import RandomForestClassifier

```

# RandomForestClassifier 실행

```
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
```



```
RandomForestClassifier(random_state=42)
```

```
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)
```

```
from sklearn.model_selection import cross_val_score

```

# 교차 검증을 통해 우리 모델이 얼마나 좋은지 알아보자

```
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```

```
0.8092759051186016
```

# SVC 사용

```
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()  # 이 모델이 더 좋아보임
```

```
0.8249313358302123
```

```
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()
```

![img](data:image/png;base64,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)

이 결과를 더욱 개선하기 위해 다음을 수행할 수 있다.

- 교차 검증 및 그리드 검색을 사용하여 더 많은 모델을 비교하고 하이퍼 파라미터를 조정한다.
- 다음과 같이 피쳐 엔지니어링을 추가로 수행한다.
- 수치 속성을 범주형 속성으로 변환해 보자. 예를 들어, 연령 그룹마다 생존율이 매우 다르므로(아래 참조) 연령 버킷 범주를 만들고 나이 대신 사용하는 데 도움이 될 수 있다. 마찬가지로, 30%만이 살아남았기 때문에 혼자 여행하는 사람들을 위해 특별한 범주를 두는 것이 유용할 수 있다(아래 참조).
- **SibSp** 및 **Parch**를 합으로 바꾼다.
- **Survived** 속성과 잘 상관되는 이름 부분을 식별한다.
- **Cabin** 열을 사용한다. 예를 들어 첫 번째 문자를 취하여 범주형 속성으로 처리한다.

```
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
```

[154]:

|           | Survived |
| --------: | -------: |
| AgeBucket |          |
|       0.0 | 0.576923 |
|      15.0 | 0.362745 |
|      30.0 | 0.423256 |
|      45.0 | 0.404494 |
|      60.0 | 0.240000 |
|      75.0 | 1.000000 |

