[enc_exp_1]: img/enc_exp_1.png

# Useful picks for machine learning engeneer

Picks about code, algorithms, pipelines etc.
___

# Table of content
[Features encoding](#Features_encoding)
    [Frequency encoding](#Frequency_encoding)
    [Order encoding](#Order_encoding)


___

# Features_encoding
## Frequency_encoding
* Encoding feature by its frequency in data :
    * вариант 1
    <X['card1_count'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))>
    * вариант 2
    <for col in ['card4', 'card6', 'ProductCD']:
        print('Encoding', col)
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        col_encoded = temp_df[col].value_counts().to_dict()   
        train_df[col] = train_df[col].map(col_encoded)
        test_df[col]  = test_df[col].map(col_encoded)
        print(col_encoded)
>
    ![Output][enc_exp_1]