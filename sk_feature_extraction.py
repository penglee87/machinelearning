from sklearn.feature_extraction import DictVectorizer

measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
    {'city': 'London', 'temperature': 13.},
]


vec = DictVectorizer()
vec_tra = vec.fit_transform(measurements)

print('vec',type(vec))
print('vec_tra',type(vec_tra))
print(type(vec_tra.toarray()))
print(vec_tra.toarray())
'''
[[  1.,   0.,   0.,  33.],
 [  0.,   1.,   0.,  12.],
 [  0.,   0.,   1.,  18.]]
'''
print(type(vec.get_feature_names()))
# <class 'list'>
print(vec.get_feature_names())
# ['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']


#############################
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
print('X',X)
print('X.toarray',X.toarray())
print('vectorizer.get_feature_names',vectorizer.get_feature_names())
# ['and', 'document', 'first', 'is', 'one','second', 'the', 'third', 'this']
print('vectorizer.vocabulary_.get',vectorizer.vocabulary_.get('first'))
# 2
print(vectorizer.transform(['Something completely new.']).toarray())

analyze = vectorizer.build_analyzer()
print('analyze1',analyze)

print('analyze2',analyze("This is a text document to analyze."))
analyze("This is a text document to analyze.") == (['this', 'is', 'text', 'document', 'to', 'analyze'])

vectorizer.get_feature_names() == (['and', 'document', 'first', 'is', 'one','second', 'the', 'third', 'this'])



