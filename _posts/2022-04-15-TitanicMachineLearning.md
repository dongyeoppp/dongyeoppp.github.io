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

```
/kaggle/input/titanic/train.csv
/kaggle/input/titanic/test.csv
/kaggle/input/titanic/gender_submission.csv
```

```
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
```

```
# 데이터를 가져와 로드한다.

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

```
# 교육세트의 상위 몇 행을 살펴보기

train_data.head()
```

| PassengerId | Survived | Pclass | Name | Sex                                               | Age    | SibSp | Parch | Ticket | Fare             | Cabin   | Embarked |      |
| :---------- | :------- | :----- | :--- | :------------------------------------------------ | :----- | :---- | :---- | :----- | :--------------- | :------ | :------- | ---- |
| 0           | 1        | 0      | 3    | Braund, Mr. Owen Harris                           | male   | 22.0  | 1     | 0      | A/5 21171        | 7.2500  | NaN      | S    |
| 1           | 2        | 1      | 1    | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0  | 1     | 0      | PC 17599         | 71.2833 | C85      | C    |
| 2           | 3        | 1      | 3    | Heikkinen, Miss. Laina                            | female | 26.0  | 0     | 0      | STON/O2. 3101282 | 7.9250  | NaN      | S    |
| 3           | 4        | 1      | 1    | Futrelle, Mrs. Jacques Heath (Lily May Peel)      | female | 35.0  | 1     | 0      | 113803           | 53.1000 | C123     | S    |
| 4           | 5        | 0      | 3    | Allen, Mr. William Henry                          | male   | 35.0  | 0     | 0      | 373450           | 8.0500  | NaN      | S    |

```
# PassengerId 열을 인덱스 열로 명시적으로 설정

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
```

```<class 'pandas.core.frame.DataFrame'>
# 누락된 데이터가 얼마나 되는지 자세히 알아보자.

train_data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 1 to 891
Data columns (total 11 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Survived  891 non-null    int64  
 1   Pclass    891 non-null    int64  
 2   Name      891 non-null    object 
 3   Sex       891 non-null    object 
 4   Age       714 non-null    float64
 5   SibSp     891 non-null    int64  
 6   Parch     891 non-null    int64  
 7   Ticket    891 non-null    object 
 8   Fare      891 non-null    float64
 9   Cabin     204 non-null    object 
 10  Embarked  889 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 83.5+ KB
```

```
train_data[train_data["Sex"]=="female"]["Age"].median()
```

Out[7]:

```
27.0
```

In [8]:

```
# 수치 속성

train_data.describe()
```

Out[8]:

|       | Survived   | Pclass     | Age        | SibSp      | Parch      | Fare       |
| :---- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- |
| count | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
| mean  | 0.383838   | 2.308642   | 29.699118  | 0.523008   | 0.381594   | 32.204208  |
| std   | 0.486592   | 0.836071   | 14.526497  | 1.102743   | 0.806057   | 49.693429  |
| min   | 0.000000   | 1.000000   | 0.420000   | 0.000000   | 0.000000   | 0.000000   |
| 25%   | 0.000000   | 2.000000   | 20.125000  | 0.000000   | 0.000000   | 7.910400   |
| 50%   | 0.000000   | 3.000000   | 28.000000  | 0.000000   | 0.000000   | 14.454200  |
| 75%   | 1.000000   | 3.000000   | 38.000000  | 1.000000   | 0.000000   | 31.000000  |
| max   | 1.000000   | 3.000000   | 80.000000  | 8.000000   | 6.000000   | 512.329200 |

```
# 목표값이 0 또는 1인지 확인

train_data["Survived"].value_counts()
```

Out[9]:

```
0    549
1    342
Name: Survived, dtype: int64
```

In [10]:

```
train_data["Pclass"].value_counts()
```

Out[10]:

```
3    491
1    216
2    184
Name: Pclass, dtype: int64
```

In [11]:

```
train_data["Sex"].value_counts()
```

Out[11]:

```
male      577
female    314
Name: Sex, dtype: int64
```

In [12]:

```
train_data["Embarked"].value_counts()
```

Out[12]:

```
S    644
C    168
Q     77
Name: Embarked, dtype: int64
```

**전처리 파이프라인 생성**

In [13]:

```
from sklearn.preprocessing import StandardScaler
```

In [14]:

```
# 이제 수치 속성을 위한 파이프라인부터 시작하여 전처리 파이프라인을 구축

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
```

In [15]:

```
# 이제 범주 속성을 위한 파이프라인을 구축

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),    
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
```

In [16]:

```
# 마지막으로 수치 및 범주형 파이프라인에 가입

from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
```

In [17]:

```
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
```

Out[17]:

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

In [18]:

```
# 라벨 받기

y_train = train_data["Survived"]
```

In [19]:

```
from sklearn.ensemble import RandomForestClassifier
```

In [20]:

```
# RandomForestClassifier 실행

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
```

Out[20]:

```
RandomForestClassifier(random_state=42)
```

In [21]:

```
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)
```

In [22]:

```
from sklearn.model_selection import cross_val_score
```

In [23]:

```
# 교차 검증을 통해 우리 모델이 얼마나 좋은지 알아보자

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```

Out[23]:

```
0.8092759051186016
```

In [24]:

```
# SVC 사용

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()  # 이 모델이 더 좋아보임
```

Out[24]:

```
0.8249313358302123
```

In [25]:

```
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()
```

![img](https://www.kaggleusercontent.com/kf/92612196/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..tUhX20hQvJFX6uppG_zxZw.WTy5Gy5TE2iYW1OssZGVfNzO-KJTJGFUfey-_Y29ttc82gV3bxgPoihFOtMqduvO4Xf_jNmOEnJiRC9oSA3SVa4vT9amqVnF52LrovHsf_kBkB82ucHKvPS3_ECnxo5Ef99A1-6Rl1dAwH7LzKpIuJT1xpUqIibtVOQMX1Y0Lt7Sul9AyUc8ZnqK3eMTGU_stCakV2Hk1D-elHINXYk0MlzVWpIkA2c__psX7BFsu27Gk9DDPreETh9zbUL_qcyXHAlOtoZAUfWWP2G7BilSzDSzOR_xusv76onyKU-kWRPcAdjmSkYtqIwhQvQYNoQH-CX6a9r8CjMOYd5w_VH8h2Ggwu2VMZk8DIyA1dqIQIlxKtKf7myIh4FndGVtpDGdslvseNprCyWkcP5ZQSbkYtEg8AQE_9zMs0Lh0czLy-5UG4P98l1M2l8WfbGruObmgv4B9I_pCcm4PdnCfQHHA8tHRxEqhzvSTpagnn9eCT8njjwjeHhOCtgzO_fFDqYvuoVrNEq4sNaHbycM7rFw3CAdiFButMPeAhuiwD1dgeyFmhTNvyB3vLBMYHaMxW-3Wf4Zkz_IA_aRv2Luidfw65B_4X7Nv7w2BCLYZyZZme4mnkEurAd1sIfMl6GsR7LFVmGoPrs30CKcnPztH-jJHzaE_BU2mfPjvs50ffujJE0.hPp-gD7J7G-6sZVbVqGt2w/__results___files/__results___25_0.png)

이 결과를 더욱 개선하기 위해 다음을 수행할 수 있다.

- 교차 검증 및 그리드 검색을 사용하여 더 많은 모델을 비교하고 하이퍼 파라미터를 조정한다.
- 다음과 같이 피쳐 엔지니어링을 추가로 수행한다.
- 수치 속성을 범주형 속성으로 변환해 보자. 예를 들어, 연령 그룹마다 생존율이 매우 다르므로(아래 참조) 연령 버킷 범주를 만들고 나이 대신 사용하는 데 도움이 될 수 있다. 마찬가지로, 30%만이 살아남았기 때문에 혼자 여행하는 사람들을 위해 특별한 범주를 두는 것이 유용할 수 있다(아래 참조).
- **SibSp** 및 **Parch**를 합으로 바꾼다.
- **Survived** 속성과 잘 상관되는 이름 부분을 식별한다.
- **Cabin** 열을 사용한다. 예를 들어 첫 번째 문자를 취하여 범주형 속성으로 처리한다.

In [26]:

```
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
```

Out[26]:

|           | Survived |
| :-------- | :------- |
| AgeBucket |          |
| 0.0       | 0.576923 |
| 15.0      | 0.362745 |
| 30.0      | 0.423256 |
| 45.0      | 0.404494 |
| 60.0      | 0.240000 |
| 75.0      | 1.000000 |

In [27]:

```
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(
    ['RelativesOnboard']).mean()
```

Out[27]:

|                  | Survived |
| :--------------- | :------- |
| RelativesOnboard |          |
| 0                | 0.303538 |
| 1                | 0.552795 |
| 2                | 0.578431 |
| 3                | 0.724138 |
| 4                | 0.200000 |
| 5                | 0.136364 |
| 6                | 0.333333 |
| 7                | 0.000000 |
| 10               | 0.000000 |