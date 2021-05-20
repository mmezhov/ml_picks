[enc_exp_1]: img/enc_exp_1.png

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

## Order encoding
Encoding by numbers

```python
df['id_38'] = df['id_38'].map({'T':1, 'F':0})
```

# NaNs treating

Calculate percentage of missing values by dataframe columns:
> to EDA library:  
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