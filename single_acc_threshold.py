# -*- coding: utf-8 -*-
"""
@Time ： 2025/1/6 15:02
@File ：single_acc_threshold.py
@IDE ：PyCharm
"""
import os

import Raster
from Judge_by_Surface_Morphology import*




def extract_stream(acc_file,out_stream_file,threshold):
    Acc = Raster.get_raster(acc_file)
    proj,geo,nodata = Raster.get_proj_geo_nodata(acc_file)

    row,col = Acc.shape
    stream = np.zeros((row,col),dtype = np.int8)

    mask = Acc >= threshold

    stream[mask] = 1

    Raster.save_raster(out_stream_file,stream,proj,geo,gdal.GDT_Byte,0)

def sbatch_extract_stream(basevenu):


    thresholds = range(450, 451, 50)


    accFile = os.path.join(basevenu,'acc.tif')
    dirFile = os.path.join(basevenu,'dir.tif')
    streamVenu = os.path.join(basevenu,'Stream3')   # 存放梯度河网阈值的结果
    if not os.path.exists(streamVenu):
        os.mkdir(streamVenu)
        os.chmod(streamVenu,0o777)
    for threshold in thresholds:

        outVenu = os.path.join(streamVenu,str(threshold))
        outstreamFile = os.path.join(outVenu,'stream.tif')
        outslinkFile = os.path.join(outVenu, 'slink.tif')
        outwatershedFile = os.path.join(outVenu, 'watershed.tif')
        if not os.path.exists(outVenu):
            os.mkdir(outVenu)
            os.chmod(outVenu,0o777)

        extract_stream(accFile,outstreamFile,threshold)
        wbt.stream_link_identifier(dirFile,outstreamFile,outslinkFile,esri_pntr=True)
        wbt.watershed(dirFile,outslinkFile,outwatershedFile,esri_pntr=True)

        print("{:s}计算完成".format(outVenu))


def calculate_drainage_density(river_raster_path, basin_raster_path, cell_size = 10):
    """
    根据栅格河网和流域计算河网密度（单位 km/km²）

    Parameters:
        river_raster_path (str): 河网栅格文件路径，河网区域像元值为1，其他为0或NoData。
        basin_raster_path (str): 流域栅格文件路径，流域内的像元有值，其他为NoData。
        cell_size (float): 像元分辨率（单位：米）。

    Returns:
        float: 河网密度（单位：km/km²）。
    """
    river_array = Raster.get_raster(river_raster_path)
    proj,geo,s_nodata = Raster.get_proj_geo_nodata(river_raster_path)
    basin_array = Raster.get_raster(basin_raster_path)
    proj,geo,b_nodata = Raster.get_proj_geo_nodata(basin_raster_path)


    # 判断有效区域
    river_array = np.where(river_array != s_nodata, 1, 0)  # 将河网像元值设置为1，其他为0
    basin_array = np.where(basin_array !=b_nodata, 1, 0)  # 将流域区域像元值设置为1，其他为0

    # 计算河流总长度
    river_length = np.sum(river_array) * cell_size / 1000  # 转换为km

    # 计算流域总面积
    basin_area = np.sum(basin_array) * (cell_size ** 2) / 1e6  # 转换为km²

    # 计算河网密度
    if basin_area == 0:
        raise ValueError("流域面积为零，无法计算河网密度！")

    drainage_density = river_length / basin_area
    return drainage_density

def sbatch_drainage_density(venu):
    import csv
    thresholds = range(500,10001,500)
    density = []
    densityfile = os.path.join(venu,'sendity.csv')
    watershedfile = os.path.join(venu,"watershed.tif")
    for threshold in thresholds:

        dirpath = os.path.join(venu,'Stream',str(threshold))
        streamfile = os.path.join(dirpath,'stream.tif')

        tempdensity = calculate_drainage_density(streamfile,watershedfile,10)
        density.append([threshold,tempdensity])

    with open(densityfile,'w') as f:
        writer = csv.writer(f)
        writer.writerows(density)
        f.close()











