>>> import numpy as np
>>> arrays = [np.random.randn(3, 4) for _ in range(10)]
>>> type(arrays)
<class 'list'>
>>> type(arrays[1])
<class 'numpy.ndarray'>
>>> arrays[1].shape
(3, 4)
>>> type(np.stack(arrays, axis=0))
<class 'numpy.ndarray'>
>>> type(np.stack(arrays, axis=0).shape)
<class 'tuple'>
>>> np.stack(arrays, axis=0).shape
(10, 3, 4)
>>> np.stack(arrays, axis=1).shape
(3, 10, 4)
>>> np.stack(arrays, axis=2).shape
(3, 4, 10)

>>> arrays = [np.random.randn(2, 3, 4) for _ in range(10)]
>>> np.stack(arrays, axis=1).shape
(2, 10, 3, 4)
>>> np.stack(arrays, axis=2).shape
(2, 3, 10, 4)
>>> np.stack(arrays, axis=3).shape
(2, 3, 4, 10)


>>> np.array([1, 2, 3]).shape  #一维数组
(3,)
>>> np.array([[1, 2, 3]]).shape  #二维数组(1*3)
(1, 3)

>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.stack((a, b))
array([[1, 2, 3],
       [2, 3, 4]])
>>> np.stack((a, b), axis=0)
array([[1, 2, 3],
       [2, 3, 4]])
>>> np.stack((a, b), axis=1)
array([[1, 2],
       [2, 3],
       [3, 4]])
>>> np.stack((a, b), axis=-1)
array([[1, 2],
       [2, 3],
       [3, 4]])
>>> np.stack((a, b), axis=-2)
array([[1, 2, 3],
       [2, 3, 4]])

>>> np.hstack((a,b))  #按顺序堆叠数组（按列方式）
array([1, 2, 3, 2, 3, 4])
>>> np.vstack((a,b))  #按顺序堆叠数组（按行方式）
array([[1, 2, 3],
       [2, 3, 4]])

>>> a = np.array([[1, 2, 3]])
>>> b = np.array([[2, 3, 4]])
>>> np.hstack((a,b))
array([[1, 2, 3, 2, 3, 4]])
>>> np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])



>>> a = np.array([[1],[2],[3]])
>>> b = np.array([[2],[3],[4]])
>>> np.hstack((a,b))  #将两个三行一列的数组横向合并
array([[1, 2],
       [2, 3],
       [3, 4]])
>>> np.vstack((a,b))  #将两个三行一列的数组纵向合并
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])
