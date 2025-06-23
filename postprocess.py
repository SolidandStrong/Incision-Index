# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/3 8:02
@Auth ： Bin Zhang
@File ：postprocess.py
@IDE ：PyCharm
"""
import os
import csv
import Raster
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt


def draw_embedding(csv_file,watershed_file,out_file):

    watershed=Raster.get_raster(watershed_file)
    proj,geo,nodata=Raster.get_proj_geo_nodata(watershed_file)

    dic={}
    with open(csv_file,'r') as f:
        csv_reader=csv.reader(f)
        n=0
        for i in csv_reader:
            if n>=1:
                dic.setdefault(int(i[0]),float(i[1]))
            n+=1
    f.close()

    row,col = watershed.shape
    arr=np.zeros((row,col),dtype=float)
    arr[:,:]=65535
    for i in range(row):
        for j in range(col):
            if watershed[i,j] != nodata:

                try:
                    value = dic[watershed[i, j]]
                    if value > 1:
                        value = 1
                    arr[i,j] = value
                except:
                    arr[i, j] = 1
    Raster.save_raster(out_file,arr,proj,geo,gdal.GDT_Float32,65535)

    # 绘制结果
    for i in range(0,7):
        threshold = i/10
        print(threshold)
        temp_arr = arr.copy()
        temp_arr[temp_arr<=threshold] = 2  # fake
        temp_arr[temp_arr<1] = 3   # ture
        out_file1=os.path.join(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\Acc300','watershed_'+str(i)+'.tif')
        Raster.save_raster(out_file1,temp_arr, proj, geo, gdal.GDT_Float32, 65535)

def check(csv_file):
    y=[]
    with open(csv_file,'r') as f:
        csv_reader=csv.reader(f)
        n=0
        for i in csv_reader:
            if n>=1:
                y.append(float(i[1]))
            n+=1
    y.sort()

    plt.hist(y,bins=100,orientation='horizontal')
    plt.scatter([i for i in range(len(y))], y,color='red',s=1)
    plt.show()

if __name__=='__main__':

    draw_embedding(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\Acc300\Acc300.csv',
                   r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\Acc300\watershed.tif',
                   r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\Acc300\New_watershed_1.tif')

    # check(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\Acc300\Acc300.csv')

    pass