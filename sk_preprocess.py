
#数据集的标准化
>>> from sklearn import preprocessing
>>> import numpy as np
>>> X_train = np.array([[ 1., -1.,  2.],
...                     [ 2.,  0.,  0.],
...                     [ 0.,  1., -1.]])

# preprocessing.scale  将原数据集转化为均值为0,标准差为1的数据集
>>> X_scaled = preprocessing.scale(X_train)
>>> X_scaled                                          
array([[ 0.  ..., -1.22...,  1.33...],
       [ 1.22...,  0.  ..., -0.26...],
       [-1.22...,  1.22..., -1.06...]])
       
>>> X_scaled.mean(axis=0)
array([ 0.,  0.,  0.])

>>> X_scaled.std(axis=0)
array([ 1.,  1.,  1.])

# preprocessing.StandardScaler() 建立标准化数据集的模型
>>> scaler = preprocessing.StandardScaler().fit(X_train)
>>> scaler
StandardScaler(copy=True, with_mean=True, with_std=True)

>>> scaler.mean_                                      
array([ 1. ...,  0. ...,  0.33...])

>>> scaler.scale_                                       
array([ 0.81...,  0.81...,  1.24...])
# 将模型应用到数据集上
>>> scaler.transform(X_train)                           
array([[ 0.  ..., -1.22...,  1.33...],
       [ 1.22...,  0.  ..., -0.26...],
       [-1.22...,  1.22..., -1.06...]])
       
       
>>> X_test = [[-1., 1., 0.]]
>>> scaler.transform(X_test)                
array([[-2.44...,  1.22..., -0.26...]])


#将特征缩放到一个范围
>>> X_train = np.array([[ 1., -1.,  2.],
...                     [ 2.,  0.,  0.],
...                     [ 0.,  1., -1.]])

# preprocessing.MinMaxScaler() 将特征缩放至[0,1]之间
>>> min_max_scaler = preprocessing.MinMaxScaler()
>>> X_train_minmax = min_max_scaler.fit_transform(X_train)
>>> X_train_minmax
array([[ 0.5       ,  0.        ,  1.        ],
       [ 1.        ,  0.5       ,  0.33333333],
       [ 0.        ,  1.        ,  0.        ]])
       
>>> X_test = np.array([[ -3., -1.,  4.]])
>>> X_test_minmax = min_max_scaler.transform(X_test)
>>> X_test_minmax
array([[-1.5       ,  0.        ,  1.66666667]])


>>> min_max_scaler.scale_                             
array([ 0.5       ,  0.5       ,  0.33...])

>>> min_max_scaler.min_                               
array([ 0.        ,  0.5       ,  0.33...])

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min


>>> X_train = np.array([[ 1., -1.,  2.],
...                     [ 2.,  0.,  0.],
...                     [ 0.,  1., -1.]])
...
>>> max_abs_scaler = preprocessing.MaxAbsScaler()
>>> X_train_maxabs = max_abs_scaler.fit_transform(X_train)
>>> X_train_maxabs                # doctest +NORMALIZE_WHITESPACE^
array([[ 0.5, -1. ,  1. ],
       [ 1. ,  0. ,  0. ],
       [ 0. ,  1. , -0.5]])
>>> X_test = np.array([[ -3., -1.,  4.]])
>>> X_test_maxabs = max_abs_scaler.transform(X_test)
>>> X_test_maxabs                 
array([[-1.5, -1. ,  2. ]])
>>> max_abs_scaler.scale_         
array([ 2.,  1.,  2.])


#缩放稀疏数据



#4.3.5。编码分类特征
>>> enc = preprocessing.OneHotEncoder()
>>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
       handle_unknown='error', n_values='auto', sparse=True)
>>> enc.transform([[0, 1, 3]]).toarray()
array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]])

#如果有可能训练数据可能缺少分类特征，则必须明确设置n_values
>>> enc = preprocessing.OneHotEncoder(n_values=[2, 3, 4])
>>> # Note that there are missing categorical values for the 2nd and 3rd
>>> # features
>>> enc.fit([[1, 2, 3], [0, 2, 0]])  
OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
       handle_unknown='error', n_values=[2, 3, 4], sparse=True)
>>> enc.transform([[1, 0, 0]]).toarray()
array([[ 0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.]])


