# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/28 17:03
@Auth ： Bin Zhang
@File ：area_elv.py
@IDE ：PyCharm
"""


from osgeo import gdal
import Raster
import numpy as np
import matplotlib.pyplot as plt

def normalization(demlist):

    maxH = max(demlist)
    minH = min(demlist)

    List = []
    for cell in demlist:
        List.append((cell-minH)/(maxH-minH))
    List.sort()
    return List

def cal_index(normalH):

    return sum(normalH)/len(normalH)

def summary_elv(watershed_file,dem_file,outfile):

    watershed = Raster.get_raster(watershed_file)
    dem = Raster.get_raster(dem_file)
    proj,geo,demnodata = Raster.get_proj_geo_nodata(dem_file)
    row,col = dem.shape

    watershedDEM = {}
    for i in range(row):
        for j in range(col):
            if dem[i,j] != demnodata:
                watershedDEM.setdefault(watershed[i,j],[]).append(dem[i,j])

    newIndex = np.zeros((row,col))
    newIndex[:,:] = -9999

    for id in watershedDEM:
        # print(watershedDEM[id])
        normalH = normalization(watershedDEM[id])

        index = cal_index(normalH)
        print(index)
        newIndex[watershed == id] = index

    Raster.save_raster(outfile,newIndex,proj,geo,gdal.GDT_Float32,-9999)




if __name__ == "__main__":

    watershed_file = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\高山区\data\watershed.tif'
    dem_file = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\高山区\data\DEM.tif'
    outfile = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\高山区\data\Tang\Tang_Result.tif'
    summary_elv(watershed_file,dem_file,outfile)

    pass