#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
https://blog.csdn.net/weiyudang11/article/details/52684333
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from  geopy.distance import great_circle
from shapely.geometry  import MultiPoint,Polygon
from geopy.geocoders  import Nominatim
from geopy.point  import Point
#import geopandas as gpd
from sklearn.preprocessing  import StandardScaler,minmax_scale


path=r"dbscan02.csv"
df=pd.read_csv(path,index_col=0,usecols=[0,1,2,3],parse_dates=[3])
#print(df.columns)
#print(df.tail(5))
#print(len(df))

df=df[(df.latitude!=0) & (df.longitude <73.3)]
print('df',len(df))
df=df.drop_duplicates()  #drop_duplicates  去重
print('df',len(df))
df_sort=df.groupby(by=df.index).count().sort_values(by="longitude",ascending=False)
print('df_sort',len(df_sort),df_sort)
dfIndex=df_sort[df_sort.longitude>30].index
dftest=df.loc[dfIndex]  #loc['index_one']：按索引选取数据
print('dftest',len(dftest),dftest.head())



##经纬度解析出城市
#def parse_city(latlng):
#    try:
#        locations=geolocator.reverse(Point(latlng),timeout=10)
#        loc=locations.raw[u'address']
#        if  u'state_district' in loc:
#            city=loc[ u'state_district'].split('/')[0]
#        else :
#            city =loc[u'county'].split('/')[0]   # 直辖市
#    except Exception as e:
#        print e
#        city=None
#    try:
#        state= loc[u'state']
#    except Exception as e:
#        print e
#        state=None
#    return city,state
#
#
#def parse_state(latlng):
#    try:
#        locations=geolocator.reverse(Point(latlng),timeout=10)
#        loc=locations.raw
#        state= loc[u'address'][u'state']
#    except Exception as e:
#        print e
#        state=None
#    return state
#
#
#geolocator = Nominatim()
#latlngs=df.ix[:,['longitude','latitude']].values
#
#df['city']=map(parse_city,latlngs) 
#
#df['state']=map(parse_state,latlngs)


#聚类分析
coords=dftest.as_matrix(columns=['longitude','latitude'])  #dataframe转matrix,df转矩阵
print('coords',len(coords),type(coords))
kms_per_radian = 6371.0088
epsilon = 10/ kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=80, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
print('cluster_labels',len(cluster_labels),type(cluster_labels))
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('clusters',len(clusters),type(clusters),clusters)
print('Number of clusters: {}'.format(num_clusters))


#类的中心点
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)
centermost_points = clusters[:-1].map(get_centermost_point)


#类中心原始数据
lons, lats = zip(*centermost_points)
rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
rs = rep_points.apply(lambda row: dftest[(dftest['latitude']==row['lat']) &(dftest['longitude']==row['lon'])].iloc[0], axis=1)


##TOPN
def getPersonlMost(dft):
    coords=dft.as_matrix(columns=['longitude','latitude'])
    kms_per_radian = 6371.0088
    epsilon = 5/ kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    clusters=pd.Series([line for line in clusters if len(line)>0])
    sorted_cluster=sorted([(get_centermost_point(line),len(line)) for line in clusters],key = lambda x:x[1],reverse=True)[:3]
    return sorted_cluster
    
    
path=r"dbscan02.csv"
#df=pd.read_csv(path,header=None,usecols=[0,1,2,3],parse_dates=[3])  #无表头时
#df.columns = ['custorm_id','latitude','longitude','create_time']
df=pd.read_csv(path,index_col=0,usecols=[0,1,2,3],parse_dates=[3])
print('headddd',df.head(5))
df=df[(df.latitude!=0) & (df.longitude < 73.3)].drop_duplicates()
df_sort=df.groupby(by=df.index).count().sort_values(by="longitude",ascending=False)
dfIndex=df_sort[df_sort.longitude>5].index
dftest=df.loc[dfIndex].dropna()
dftest.to_csv("deftest.csv")


TopN=[]
dftest=pd.read_csv("deftest.csv",index_col=0).drop_duplicates().dropna()
cnt=0
for line in dftest.index.unique():
    cnt+=1
    if cnt%1000==0:
        print (cnt)
    dfs=dftest.loc[line]
    cc=getPersonlMost(dfs)

    TopN.append(cc)
peronalMost=pd.DataFrame(TopN,index=dftest.index.unique(),columns=["mostly",'secondly','merely'])
peronalMost.to_csv("personal_most.csv")
