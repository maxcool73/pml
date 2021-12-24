import pandas as pd
from io import StringIO

csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
df
df.isnull().sum()
df.values
df.dropna(axis=0)
df.dropna(axis=1)
df.dropna(how='all')
df.dropna(subset=['C'])

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data
df.fillna(df.mean())

df = pd.DataFrame([
    ['green', 'М', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['Ыuе', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
df
size_mapping = {'XL': 3,
                'L': 2,
                'М': 1}
df['size'] = df['size'].map(size_mapping)
df
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)
class_mapping = {label: idx for idx, label in
                 enumerate(np.unique(df['classlabel']))}
class_mapping
df['classlabel'] = df['classlabel'].map(class_mapping)
df
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df
