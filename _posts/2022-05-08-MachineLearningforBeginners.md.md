# MachineLearning-for-Beginners



## 분류 (요리 데이터 세트 사례를 통해 머신러닝을 알아가보자)

우리가 이 요리 데이터 세트에 대해 묻고 싶은 질문은 사실 다종다양한 질문입니다. 우리는 몇 가지 잠재적 국가 요리를 다룰 수 있기 때문입니다. 성분 배치가 주어졌을 때, 이 많은 등급 중 어떤 데이터가 적합할까요?

Scikit-learn은 해결하려는 문제의 종류에 따라 데이터를 분류하는 데 사용할 수 있는 몇 가지 다른 알고리즘을 제공합니다. 다음 두 가지 수업에서는 이러한 알고리즘 중 몇 가지에 대해 배우게 됩니다.

## 연습 - 데이터 정리 및 균형 조정

이 프로젝트를 시작하기 전에 가장 먼저 해야 할 일은 데이터를 정리하고 균형을 맞춰 더 나은 결과를 얻는 것입니다.

가장 먼저 설치해야 할것은  [imblearn](https://imbalanced-learn.org/stable/)이다.  이 패키지는 데이터의 균형을 더 잘 조정할 수 있도록 지원하는 사이킷런 패키지이다

1. `imblearn`을 설치하려면 다음과 같이 `pip install`를 실행한다:

    ```python
    pip install imblearn
    ```

2. 데이터를 가져오는 데 필요한 패키지를 가져와 시각화하고, '`mblearn`에서 `SMOTE`도 가져온다.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    이제 다음 데이터를 읽을 준비가 되었다.

3. 다음 작업은 데이터를 가져오는 것이다:

   ```python
   df  = pd.read_csv('../data/cuisines.csv')
   ```

   `read_csv()`를 사용하면 csv 파일 _cusines.csv_의 내용을 읽고 변수 `df`에 배치한다.

4. 데이터의 형태 확인:

   ```python
   df.head()
   ```

   첫 다섯줄을 다음과 같이 확인할 수 있다:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

5. `info()`를 호출하여 이 데이터에 대한 정보 가져오기`:

    ```python
    df.info()
    ```

    확인:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## 운동 - 요리에 대한 학습

이제 그 일은 더 흥미로워지기 시작한다. 요리 당 데이터의 분포를 알아보겠습니다.

1. `barh()`를 호출하여 데이터를 막대로 표시합니다:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![cuisine-dist](https://user-images.githubusercontent.com/103740881/167290106-e1ebaee8-2935-4bfe-ace5-f6f6ef33d46e.png)

    한정된 수의 요리가 있지만, 데이터의 분포가 고르지 않다. 우린 이것을 고칠 수 있다. 수정하기 전 조금 더 살펴보자.

1. 요리당 얼마나 많은 데이터를 사용할 수 있는지 확인하고 출력한다:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    출력 확인:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## 성분 발견

이제 여러분은 데이터를 더 깊이 파고들어 요리당 전형적인 재료가 무엇인지 배울 수 있습니다. 음식 사이에 혼란을 일으키는 반복적인 데이터를 지워야 하는데, 이 문제에 대해 알아보도록 하자.

1. Python에서 `create_ingredient()` 함수를 만들어 성분 데이터 프레임을 생성한다. 이 기능은 도움이 되지 않는 열을 떨어뜨리는 것으로 시작하고 성분 수를 기준으로 재료를 정렬합니다.:

   ```python
   def create_ingredient_df(df):
       ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
       ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
       ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
       inplace=False)
       return ingredient_df
   ```

   이제 그 기능을 사용하여 요리별로 가장 인기 있는 10대 식재료에 대한 아이디어를 얻을 수 있습니다.

2. 'create_ingredient()'를 호출하고 barh()를 호출하여 플롯합니다.:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](https://user-images.githubusercontent.com/103740881/167290374-d83a284f-2dbe-4a62-b10f-08116d2150de.png)

3. 일본 데이터에 대한 동일한 작업 수행:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](https://user-images.githubusercontent.com/103740881/167290424-21a84dcb-3380-47ef-9655-908c74ecbe47.png)

4. 중국 재료:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](https://user-images.githubusercontent.com/103740881/167290477-8588fc93-f864-4301-b2ef-ca09889b0415.png)

5. 인도재료를 플롯한다:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](https://user-images.githubusercontent.com/103740881/167290480-6284de90-e2dd-435c-a2c6-86c8fdbdaac7.png)

6. 마지막으로 한국 재료를 플롯:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](https://user-images.githubusercontent.com/103740881/167290483-ba0fac7f-c598-4978-93c3-df7c769dcad5.png)

7. 이제, `drop()`을 불러서, 구별되는 요리들 사이에 혼란을 일으키는 가장 일반적인 재료들을 버려라:

   모든 사람들은 쌀, 마늘, 생강을 좋아합니다!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## 데이터세트 균형

이제 데이터를 정리했으므로`SMOTE`를  사용하여 균형을 잡는다.

1. 새로운 샘플을 생성하는 전략인 `fit_resample()`을 호출한다.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    데이터의 균형을 유지함으로써 데이터를 분류할 때 더 나은 결과를 얻을 수 있습니다. 이진 분류에 대해 생각해 보세요. 대부분의 데이터가 하나의 클래스인 경우 ML 모델은 해당 클래스를 더 자주 예측합니다. 단지 더 많은 데이터가 있기 때문입니다. 데이터의 균형을 맞추려면 왜곡된 데이터가 필요하며 이러한 불균형을 제거하는 데 도움이 됩니다.

1. 이제 성분 당 라벨 수를 확인할 수 있습니다:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    출력 확인:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    데이터는 멋지고 깨끗하고, 균형이 잡혀 있다.

1. 마지막 단계는 레이블과 기능을 포함한 균형 잡힌 데이터를 파일로 내보낼 수 있는 새로운 데이터 프레임에 저장하는 것입니다.:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. `transformed_df.head()`와 `transformed_df.info()`를 사용하여 데이터를 한 번 더 살펴볼 수 있습니다. 향후 학습에 사용할 수 있도록 이 데이터 사본 저장:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    이제 루트 데이터 폴더에서 이 새 CSV를 찾을 수 있습니다.



# 요리 분류기 1

이 과정에서는 지난 수업에서 저장한 모든 음식에 대한 균형 잡힌 깨끗한 데이터로 가득 찬 데이터 세트를 사용합니다.

이 데이터 세트를 다양한 분류기와 함께 사용하여 재료 그룹을 기반으로 특정 국가 음식을 예측합니다. 이렇게 하는 동안 분류 작업에 알고리즘을 활용할 수 있는 몇 가지 방법에 대해 자세히 알아볼 수 있습니다.



# 준비

레슨 1을 완료했다고 가정하면 이 네 가지 레슨에 대한 루트 데이터 폴더에  cleaned_cuisines.csv 파일이 있는지 확인합니다.

## 연습 - 국가 음식 예측

1.  notebook.ipynb 폴더에서 작업할 때, Pandas 라이브러리와 함께 해당 파일을 가져옵니다.:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    데이터 확인:

|      | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ...  | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam  | yeast | yogurt | zucchini |
| ---- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | ---- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | ---- | ----- | ------ | -------- |
| 0    | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ...  | 0       | 0           | 0          | 0                       | 0    | 0    | 0    | 0     | 0      | 0        |
| 1    | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ...  | 0       | 0           | 0          | 0                       | 0    | 0    | 0    | 0     | 0      | 0        |
| 2    | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ...  | 0       | 0           | 0          | 0                       | 0    | 0    | 0    | 0     | 0      | 0        |
| 3    | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ...  | 0       | 0           | 0          | 0                       | 0    | 0    | 0    | 0     | 0      | 0        |
| 4    | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ...  | 0       | 0           | 0          | 0                       | 0    | 0    | 0    | 0     | 1      | 0        |


1. 이제 라이브러리를 여러 개 더 가져옵니다:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. 훈련을 위해 X 및 Y 좌표를 두 개의 데이터 프레임으로 나눕니다. `cuisine` 라벨 데이터 프레임이 될 수 있습니다.:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    확인:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1.  `Unnamed: 0` 열과 `drop()`이라고 하는 `cuisine`열을 삭제한다. 나머지 데이터를 훈련 가능한 기능으로 저장:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    특징 확인:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 |

이제 모델을 훈련할 준비가 되었다!

## 분류자 선택

데이터가 깨끗해지고, 훈련 준비가 되었으므로 작업에 사용할 **알고리즘**을 결정해야 한다.

`Scikit-learn` 그룹은 **지도 학습**에서 분류되며, 이 범주에서 분류할 수 있는 다양한 방법을 찾을 수 있다.

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)



### 어떤 분류기로 해야 할까?

여러 개를 훑어보고 좋은 결과를 찾는 것이 테스트하는 방법입니다. 위에 기재된 표현방법들을 비교해보고 그 결과들을 시각화하여 보자.

![comparison](https://user-images.githubusercontent.com/103740881/167291729-d721dbe0-c6e6-422d-88ee-b408ad3fb73b.png)

## 더 나은 접근

함부로 추측하는 것보다 더 나은 방법이다. 여기서, 우리는 다중 클래스 문제에 대해 몇 가지 선택사항이 있다는 것을 발견한다.

<img width="364" alt="cheatsheet" src="https://user-images.githubusercontent.com/103740881/167291847-831a21b2-7501-4216-99ab-0cc02f05b9fc.png">	

### 추리

우리가 가지고 있는 제약 조건들을 고려할 때, 우리가 다른 접근 방식들을 통해 우리의 길을 추론할 수 있는지 알아봅시다.

- 깨끗하지만 최소한의 데이터 세트와 노트북을 통해 로컬로 교육을 실행하고 있다는 사실을 감안할 때 신경망은 이 작업에 너무 무겁다.
- 우리는 2등급 분류기를 사용하지 않기 때문에 one-vs-all을 배제한다.
- 의사결정 트리가 작동하거나 다중 클래스 데이터에 대해 로지스틱 회귀 분석을 수행할 수 있습니다.
- 멀티클래스 부스트 결정 트리는 예를 들어 순위를 구축하도록 설계된 작업과 같은 비모수 작업에 가장 적합하므로 우리에게 유용하지 않다.

### 사이킷런 사용

우리는 Scikit-learn을 사용하여 데이터를 분석할 것이다. 그러나 Scikit-learn에서는 로지스틱 회귀 분석을 사용하는 여러 가지 방법이 있습니다.  

본적으로 Scikit-learn에게 로지스틱 회귀 분석을 수행하도록 요청할 때 지정해야 하는 두 가지 중요한 매개 변수인 'multi_class'와 'solver'가 있다. 'multi_class' 값은 특정 동작을 적용합니다. 해결사의 값은 사용할 알고리즘입니다. 모든 solver를 모든 'multi_class' 값과 쌍으로 구성할 수는 없습니다.

문서에 따르면, 멀티클래스 사례에서 훈련 알고리즘은 다음과 같다.

- **one-vs-rest (OvR) 체계사용**
- **cross-entropy loss 사용**

Scikit-learn은 해결사가 다양한 종류의 데이터 구조에서 나타나는 다양한 문제를 처리하는 방법을 설명하는 이 표를 제공합니다.

<img width="765" alt="solvers" src="https://user-images.githubusercontent.com/103740881/167292092-dba83317-5997-4267-b6e0-3fde722a7af4.png">

## 연습 - 데이터 분할

최근 이전 수업에서 후자에 대해 알게 된 이후 첫 번째 교육 시행을 위해 로지스틱 회귀 분석에 초점을 맞출 수 있습니다.
train_test_split()을 호출하여 데이터를 교육 및 테스트 그룹으로 나눕니다.

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 연습 - 로지스틱 회귀 분석 적용

멀티클래스 케이스를 사용 중이므로 사용할 _scheme_와 설정할 _solver_를 선택해야 합니다. 다중 클래스 설정 및 **liblinear** solver과 함께 로지스틱 회귀 분석을 사용하여 훈련합니다.

1. multi_class를 'ovr'로 설정하고 solver를 'liblinear'로 설정하여 로지스틱 회귀 분석 생성:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    정확도는 **80% 이상**!

2. 하나의 데이터 행(#50)을 테스트하면 이 모델이 작동하는 것을 볼 수 있습니다:

   ```python
   print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
   print(f'cuisine: {y_test.iloc[50]}')
   ```

   결과 확인:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

1. 더 깊이 파고들면 이 예측의 정확성을 확인할 수 있습니다:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    결과가 인쇄됩니다. 인도 요리가 가장 잘 추측됩니다. 그럴 확률이 높습니다.:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

4. 회귀 분석 레슨에서 했던 것처럼 분류 보고서를 인쇄하여 더 자세히 알아보기:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |



# 요리 분석기 2

이 두 번째 분류 과정에서는 숫자 데이터를 분류하는 더 많은 방법을 살펴보겠습니다. 또한 한 분류기를 다른 분류기로 선택하는 데 미치는 영향에 대해서도 배울 수 있습니다.



### 전제 조건

이전 레슨을 완료했으며 이 4레슨 폴더의 루트에 _cleaned_cuisines.csv_라는 'data' 폴더에 정리된 데이터 세트가 있다고 가정합니다.

### 준비

_notebook.ipynb_ 파일을 정리된 데이터 세트로 로드하고 모델 구축 프로세스를 위해 X 및 y 데이터 프레임으로 분할했습니다.

## 분류 지도

이전에는 Microsoft의 치트 시트를 사용하여 데이터를 분류할 때 사용할 수 있는 다양한 옵션에 대해 배웠습니다. Scikit-learn은 추정기를 더 좁히는 데 도움이 될 수 있는 유사하지만 보다 세분화된 치트 시트를 제공합니다(분류기의 다른 용어).

![map](https://user-images.githubusercontent.com/103740881/167292449-2839d329-29f7-43ea-af94-58243d8d5433.png)

### 계획

이 지도는 데이터를 명확하게 파악한 후 의사 결정에 이르는 경로를 '걷기'할 수 있으므로 매우 유용합니다:

- 우리는 50개 이상의 샘플을 가지고 있다.
- 우리는 카테고리를 예측하길 원한다.
- 우리는 라벨 데이터를 갖고 있다.
- 우리는 10만개 미만의 샘플을 가지고 있습니다.
- 우리는 Linear SVC를 고를 수 있다.
- 만약 그게 안 된다면, 우리는 수치 데이터를 가지고 있기 때문에
    -  KNeighbors Classifier을 시도할 수 있다. 
      - 작동하지 않는다면 SVC ,Ensemble Classifiers 시도

이것은 따라가기에 매우 도움이 되는 길이다.

## 연습 - 데이터 분할

이 경로를 따라 사용할 라이브러리를 가져오는 것부터 시작해야 합니다.

1. 필요한 라이브러리 가져오기:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1.  훈련 및 테스트 데이터 분할:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## 선형 SVC 분류기

SSVC(support-Vector clustering)는 ML 기술인 Support-Vector machine 제품군의 하위 제품이다. 이 방법에서는 '커널'을 선택하여 레이블을 군집화하는 방법을 결정할 수 있습니다. 'C' 파라미터는 파라미터의 영향을 조절하는 '정규화'를 의미한다.

### 연습 - 선형 SVC 적용

classifiers배열을 만드는 것으로 시작합니다. 테스트하는 대로 이 어레이에 점진적으로 추가합니다.

1. 선형의 SVC :

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. 선형 SVC를 사용하여 모델 교육 및 보고서 인쇄:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    결과가 좋다:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-Neighbors 분류

K-Neighbors는 ML 방법의 "Neighbors" 계열의 일부로, 지도 학습과 비지도 학습 모두에 사용할 수 있다. 이 방법에서는 데이터에 대한 일반화된 레이블을 예측할 수 있도록 미리 정의된 수의 점이 생성되고 이러한 점 주위에 데이터가 수집됩니다.

### 연습 - K-Neighbors 분류기 적용

이전의 분류기는 좋았고, 데이터도 잘 작동했지만, 아마도 우리는 더 나은 정확도를 얻을 수 있을 것이다. K-Neighbors 분류기를 사용해 보십시오.

1. classifier배열에 줄 추가 (선형 SVC 항목 뒤에 쉼표 추가):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    결과는 조금 더 나쁘다:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```


## 지원 벡터 분류기

Support-Vector 분류기는 분류 및 회귀 작업에 사용됩니다. SVM은 두 범주 간의 거리를 최대화하기 위해 "공간 내 지점에 교육 예제를 매핑"합니다. 후속 데이터는 해당 범주를 예측할 수 있도록 이 공간에 매핑됩니다.

### 연습 - 지원 벡터 분류기 적용

Support Vector Classifier(지원 벡터 분류기)를 사용하여 더 나은 정확도를 위해 노력해 보겠습니다.

1. K-Neighbors 항목 뒤에 쉼표를 추가한 다음 이 줄을 추가합니다.:

    ```python
    'SVC': SVC(),
    ```

    결과가 꽤 좋다.

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```


## 분류자 집합

지난번 시험은 꽤 잘 봤지만 끝까지 그 길을 따라가자. 앙상블 분류기, 특히 랜덤 포레스트와 AdaBoost를 사용해 봅시다.

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

특히 랜덤 포레스트의 경우 결과가 매우 우수합니다:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```



## 모델 구축

응용 ML 시스템을 구축하는 것은 비즈니스 시스템에 이러한 기술을 활용하는 데 있어 중요한 부분입니다. Onnx를 사용하여 웹 응용 프로그램 내에서 모델을 사용할 수 있으므로 필요한 경우 오프라인 컨텍스트에서 모델을 사용할 수 있습니다.

이 과정에서는 추론을 위한 기본 JavaScript 기반 시스템을 구축할 수 있습니다. 그러나 먼저 모델을 교육하고 Onnx에서 사용하도록 변환해야 합니다.

## 연습 - 열차 분류 모델

첫째, 우리가 사용한 청정 요리 데이터 세트를 사용하여 분류 모델을 교육한다. 

1. 유용한 라이브러리 가져오기 시작:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

1. 그런 다음 _read_csv()_를 사용하여 CSV 파일을 읽으면서 이전 수업과 동일한 방식으로 데이터를 처리합니다`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. 처음 두 개의 불필요한 열을 제거하고 나머지 데이터를 'X'로 저장합니다:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. 레이블을 'y'로 저장:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### 교육 루틴을 시작합니다.

우리는 정확도가 좋은 'SVC' 라이브러리를 사용할 것이다.

1. Scikit에서 적절한 라이브러리 가져오기 - 학습:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. 별도의 교육 및 테스트 세트:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. 이전 수업과 마찬가지로 SVC 분류 모델 구축:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. 이제 _predict()_를 호출하여 모델을 테스트합니다.:

    ```python
    y_pred = model.predict(X_test)
    ```

1. 모델의 품질을 확인하기 위한 분류 보고서 출력:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    우리가 전에 봤듯이 정확도가 좋다:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### 모델을 Onnx로 변환

반드시 적절한 텐서 수로 변환해야 한다. 이 데이터 집합에는 380개의 성분이 나열되어 있으므로 _FloatTensorType_에서 이 숫자를 확인해야 합니다.

1. 380의 tensor 수를 사용하여 변환.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. onx를 생성하고 파일 **model.onnx**로 저장합니다:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```


전체 노트북을 실행하면 이제 Onnx 모델이 구축되어 이 폴더에 저장됩니다.

## 모델 보기

Onnx 모델은 Visual Studio 코드에서 잘 보이지 않지만, 많은 연구자들이 모델을 시각화하기 위해 사용하는 매우 좋은 무료 소프트웨어가 있습니다:

<img width="196" alt="netron" src="https://user-images.githubusercontent.com/103740881/167293544-c9d16f08-b382-4d5a-8129-a0a10c0c4f9a.png" style="zoom: 150%;" >

Netron은 모델을 보는 데 유용한 도구입니다.

이제 웹 앱에서 이 깔끔한 모델을 사용할 준비가 되었습니다. 여러분이 냉장고 안을 볼 때 유용하게 사용할 수 있는 앱을 만들고, 여러분의 모델에 의해 결정되는 대로, 여러분이 주어진 요리를 요리하기 위해 어떤 남은 재료의 조합을 사용할 수 있는지 알아봅시다.

## 추천자 웹 응용 프로그램 구축

웹 앱에서 직접 모델을 사용할 수 있습니다. 또한 이 아키텍처를 통해 필요한 경우 로컬 및 오프라인에서도 실행할 수 있습니다. 먼저 `model.onnx` 파일을 저장한 폴더와 동일한 폴더에 `index.html` 파일을 만듭니다.

1. 이 파일 _index.html_에서 다음 마크업을 추가하십시오:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. 이제 'body' 태그 내에서 작업하면서 일부 성분을 반영하는 확인란 목록을 표시하기 위해 약간의 마크업을 추가합니다:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    각 확인란에는 값이 지정됩니다. 이는 데이터 세트에 따라 성분이 발견되는 지수를 반영한다. 예를 들어, 이 알파벳 목록에서 Apple은 다섯 번째 열을 차지하기 때문에, 0에서 숫자를 세기 시작할 때 값은 '4'입니다.

1.  [Onnx Runtime](https://www.onnxruntime.ai/) 가져오기:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    

1. 런타임이 설치되면 런타임을 호출할 수 있습니다:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });
    
        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }
    
        async function startInference() {
    
            let atLeastOneChecked = testCheckboxes()
    
            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');
    
                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };
    
                // feed inputs and run
                const results = await session.run(feeds);
    
                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')
    
            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

이 코드에는 다음과 같은 몇 가지 일이 발생합니다:

1. 성분 확인란이 선택되었는지 여부에 따라 380개의 가능한 값(1 또는 0)을 설정하여 모형으로 전송하여 추론했습니다.
2. 확인란 배열과 응용 프로그램이 시작될 때 호출되는 `init` 함수에서 확인되었는지 확인하는 방법을 만들었습니다. 확인란을 선택하면 선택한 성분을 반영하도록 성분 배열이 변경됩니다.
3. 확인란이 선택되어 있는지 확인하는 `test Checkboxes` 함수를 만들었습니다.
4.  버튼을 누르면`startInference` 기능을 사용하고, 체크박스를 켜면 추론을 시작합니다.
5. 추론 루틴에는 다음이 포함됩니다:
   1. 모델의 비동기 부하 설정
   2. 모델에 보낼 텐서 구조 생성
   3. 모델을 교육할 때 작성한 `float_input` 입력을 반영하는 'feeds' 생성(Netron을 사용하여 해당 이름을 확인할 수 있음)
   4. 이러한 'feeds'를 모델에 보내고 응답을 기다리는 중



## 응용 프로그램 테스트

index.html 파일이 있는 폴더에서 Visual Studio Code에서 터미널 세션을 엽니다. [http-server](https://www.npmjs.com/package/http-server)가 전체적으로 설치되어 있는지 확인하고 프롬프트에 `http-server`를 입력합니다. 로컬 호스트가 열리면 웹 앱을 볼 수 있습니다. 다양한 재료에 따라 어떤 요리를 추천하는지 확인합니다.



<img width="816" alt="web-app" src="https://user-images.githubusercontent.com/103740881/167294305-d295d735-6163-4ec1-8304-2506bfaa50ea.png">

축하합니다, 몇 개의 필드가 있는 'recommendation' 웹 앱을 만들었습니다. 이 시스템을 구축하기 위한 시간을 가지세요!