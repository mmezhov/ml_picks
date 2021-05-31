[enc_exp_1]: img/enc_exp_1.png
[MetricVisualizerPlot]: img/MetricVisualizerPlot.png

# Useful picks for machine learning engeneer

Picks about code, algorithms, pipelines etc.

# Table of content
- [Features encoding](#Features-encoding)  
    - [Frequency encoding](#Frequency-encoding)  
    - [Order encoding](#Order-encoding)  
- [NaNs treating](#NaNs-treating)

___

# Features encoding
## Frequency encoding
Encoding feature by its frequency in data :  
variant 1  
```python
X['card1_count'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
```
variant 2  
```python
for col in ['card4', 'card6', 'ProductCD']:
    print('Encoding', col)
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    col_encoded = temp_df[col].value_counts().to_dict()   
    train_df[col] = train_df[col].map(col_encoded)
    test_df[col]  = test_df[col].map(col_encoded)
    print(col_encoded)
```
![Output][enc_exp_1]

## Ordinal encoding
The features are converted to ordinal integers. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)  

variant 1  
```python
df['id_38'] = df['id_38'].map({'T':1, 'F':0})
```

variant 2  
```python

```

## Combinations
itertools.combinations(iterable, [r]),  
[source](https://docs.python.org/3/library/itertools.html#itertools.combinations)  
```python
>>> itertools.combinations('ABCD', 2)
AB AC AD BC BD CD
```
> 

# NaNs treating

Calculate percentage of missing values by dataframe columns:
> to EDA module:  
```python
def get_percentage_of_NaNs(df):
    """
    Return dataframe with list of column names and percentage of missing
    values of each column.
    :param df: input dataframe
    """
    percent_of_missings = df.isnull().sum()*100/len(df)
    missings = pd.DataFrame({'column name': df.columns,
                             'percent of missings': percent_of_missings}, 
                            index=None)
    missings = missings.reset_index().drop('index', axis=1)
    return missings
```

Visualise features on bar plot, where axis Y - is a fraction of missings values:  
```python
(train_data.isnull().sum()/len(train_data)).plot(kind='bar', figsize=(15,7))
```

# Filtering and fetching data

Fetching categorical feature with `str` and regular expression:
```python
>>> s = pd.Series(["1+1=2"])
>>> s
0    1+1=2
dtype: object
>>> s.str.split(r"\+|=", expand=True)
     0    1    2
0    1    1    2
```
[original from documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html)


# CatBoost tips

[Masterclass about solving classification problem with catboost / Мастер класс Решение задач классификации при помощи CatBoost – Никита Дмитриев](https://github.com/catboost/catboost/blob/master/catboost/tutorials/events/pydata_moscow_oct_13_2018.ipynb)


Visualize training process with plots of metrics:  
1 step - using `train_dir` param, ex. `my_baseline_model` in constructor of model:  
```python
model = CatBoostClassifier(
    train_dir='my_baseline_model'
)
```  
2 step v1 - import and use catboost build in visualizer (`!works not in all cases`):  
```python
from catboost import MetricVisualizer

MetricVisualizer(['my_baseline_model']).start()
```  
![MetricVisualizer plot][MetricVisualizerPlot]  
Probably will need to run following command before running Juypter Notebook
```
jupyter nbextension enable --py widgetsnbextention
```

2 step v2 - using tensorboard ([more information](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks)):  
a) terminal  
```
>>> tensorboard --logdir=<train_dir>
```
b) jupyter notebook  
[example](https://github.com/tensorflow/tensorboard/blob/master/docs/get_started.ipynb)  
```python
# Load the TensorBoard notebook extension
%load_ext tensorboard

%tensorboard --logdir <train_dir>
```


# Date-time picks

```python
# day of week
df['dow'] = df['created'].apply(lambda x: x.date().weekday())
df['is_weekend'] = df['created'].apply(lambda x: 1 if x.date().weekday() in (5, 6) else 0)
```

# Correlation

Cross-correlation of two 1-dimensional sequences:  
[source](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html)
```python
>>> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
array([0.5,  2. ,  3.5,  3. ,  0. ])
```