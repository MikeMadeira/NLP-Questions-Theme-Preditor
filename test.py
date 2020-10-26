from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)

print(pipe['count'].transform(corpus).toarray())

print(pipe['tfid'].idf_)

tfidf = pipe.transform(corpus).toarray()
print(pipe['count'].get_feature_names())
print(tfidf)

df = pd.DataFrame(tfidf)
df.columns = vocabulary

# print(df.append(df.sum(axis=0), ignore_index=True) )

print([remove_column for remove_column in df.columns if df[remove_column].sum() < 0.6])