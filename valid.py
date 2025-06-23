# -*- coding: utf-8 -*-
"""
@Time ： 2025/1/6 16:31
@File ：valid.py
@IDE ：PyCharm
"""
import numpy as np
import os
import Raster
import csv
from osgeo import gdal

def error_matrix(trueStreamFile,streamFile,oringinStreamfile):

    """
    将prun后的河网与NHD河网进行匹配，记录各分类类别的数量，构建误差矩阵.
    1）若河流cell数量落在某河段超过10个，则认为是重合，TP；

    :param trueStreamFile:
    :param streamFile:
    :param outFile:
    :return:
    """
    oringinStream = Raster.get_raster(oringinStreamfile)
    _,_,onodata = Raster.get_proj_geo_nodata(oringinStreamfile)
    Tstream = Raster.get_raster(trueStreamFile)
    stream = Raster.get_raster(streamFile)
    proj,geo,tnodata = Raster.get_proj_geo_nodata(trueStreamFile)
    _,_,s_nodata = Raster.get_proj_geo_nodata(streamFile)
    row,col = Tstream.shape


    # F3、河网重叠条数
    TP = 0  # 落在NHD上的条数
    redundant = set()  # 未落在NHD的冗余条数
    notcheck = set()  # 未落在NHD的被剔除的条数


    maskNHD =  Tstream.copy()
    maskNHD[maskNHD!=tnodata] = stream[maskNHD!=tnodata]
    # maskNHD[maskNHD==tnodata] = 0
    maskStream = stream.copy()
    # maskStream[maskStream==s_nodata] = 0
    maskOStream = oringinStream.copy()
    originStreamNum = len(np.unique(maskOStream[maskStream!=onodata]))   # 一个重要的参数：初始河网数量

    streamNum = len(np.unique(maskStream[maskStream!=s_nodata]))
    rivers = {}
    orivers = {}
    for i in range(row):
        for j in range(col):
            if stream[i,j] != s_nodata:
                rivers.setdefault(stream[i,j],[]).append((i,j))
            if oringinStream[i,j] != onodata:
                orivers.setdefault(oringinStream[i,j],[]).append((i,j))

    # 计算初始冗余河网
    originIds = set()
    OTPs = 0
    for s_id in orivers:
        cells = orivers[s_id]
        flag = False
        for cell in cells:
            if maskNHD[cell[0], cell[1]] != tnodata:
                OTPs += 1
                originIds.add(maskOStream[cell[0], cell[1]])
                flag = True
                break
        if not flag:
            continue
        for cell in cells:
            maskOStream[cell[0],cell[1]] = onodata
            maskNHD[cell[0], cell[1]] = tnodata
            # maskStream[cell[0], cell[1]] = s_nodata
    # 剔除已经计算过的较长NHD河网
    for NHDid in originIds:
        maskNHD[maskNHD==NHDid] = tnodata

    # print(originStreamNum,OTPs,originStreamNum-OTPs)
    onotcheck = len(np.unique(maskNHD[maskNHD!=tnodata]))
    oredundant = len(np.unique(maskOStream[maskOStream != onodata]))

    # calculate TP，与标准NHD河网有交集及为TP，要删除TP，剩余的即为冗余和未检测
    maskStream = stream.copy()
    maskNHD = Tstream.copy()

    TPNHDids = set()
    for s_id in rivers:
        cells = rivers[s_id]
        flag = False
        for cell in cells:
            if maskNHD[cell[0],cell[1]] != tnodata:
                TP += 1
                TPNHDids.add(maskNHD[cell[0],cell[1]])
                flag = True
                break
        if not flag:
            continue
        for cell in cells:
            maskNHD[cell[0],cell[1]] = tnodata
            maskStream[cell[0],cell[1]] = s_nodata
    # 剔除已经计算过的较长NHD河网
    for NHDid in TPNHDids:
        maskNHD[maskNHD==NHDid] = tnodata

    redundant = len(np.unique(maskStream[maskStream!=s_nodata]))
    notcheck = len(np.unique(maskNHD[maskNHD!=tnodata]))

    NHDNum = TP + notcheck

    print([streamNum, NHDNum, TP, redundant, notcheck,oredundant,onotcheck])  # All = sNum + NNum + nC
    return [streamNum, NHDNum, TP, redundant, notcheck,originStreamNum,OTPs,oredundant,onotcheck]


    #
    # for i in range(row):
    #     for j in range(col):
    #         if stream[i, j] == s_nodata and Tstream[i, j] == tnodata:
    #             continue
    #         if stream[i,j] != s_nodata:
    #             streamNum.add(stream[i,j])
    #         if Tstream[i,j] != tnodata:
    #             NHDNum.add(Tstream[i,j])
    #         if stream[i,j] != s_nodata and Tstream[i,j] != tnodata:
    #             TP.add(Tstream[i,j])
    #         if stream[i,j] != s_nodata and Tstream[i,j] == tnodata: # 冗余
    #             redundant.add(stream[i,j])
    #         if stream[i, j] == s_nodata and Tstream[i, j] != tnodata:  # 未检测到
    #             notcheck.add(Tstream[i,j])
    #
    # return [streamArea,NHDArea,TP,redundant,notcheck]

    # # F1、河网重叠面积
    # streamArea = 0
    # NHDArea = 0
    # TP = 0  # 落在NHD上的面积
    # redundant = 0  # 未落在NHD的冗余面积
    # notcheck = 0  # 未落在NHD的被剔除的面积
    # for i in range(row):
    #     for j in range(col):
    #         if stream[i, j] == s_nodata and Tstream[i, j] == tnodata:
    #             continue
    #         if stream[i,j] != s_nodata:
    #             streamArea += 1
    #         if Tstream[i,j] != tnodata:
    #             NHDArea += 1
    #         if stream[i,j] != s_nodata and Tstream[i,j] != tnodata:
    #             TP += 1
    #         if stream[i,j] != s_nodata and Tstream[i,j] == tnodata:
    #             redundant += 1
    #         if stream[i, j] == s_nodata and Tstream[i, j] != tnodata:
    #             notcheck += 1
    # return [streamArea,NHDArea,TP,redundant,notcheck]




    # F2、误差矩阵
    # sid = {}
    # allNum = set()  # 所有的被验证河段
    # for i in range(row):
    #     for j in range(col):
    #         if stream[i,j] == s_nodata:
    #             continue
    #         if Tstream[i,j] == tnodata:
    #             continue
    #         allNum.add(stream[i,j])
    #         sid.setdefault(stream[i,j],{}).setdefault(Tstream[i,j],[]).append(1)
    #
    #
    # allStream = len(allNum)
    # Tids = set()   # TP 河段
    # for s_id in sid:
    #     for Tid in sid[s_id]:
    #         if len(sid[s_id][Tid]) >= 3:
    #             Tids.add(sid)
    # TP = len(Tids)
    # NHDNum = np.max(Tstream[Tstream!=tnodata]) + 1
    # FN = int(NHDNum)
    # FP = allStream-TP
    # TN = FP - FP
    # # print(allNum,NHDNum)
    # # print(TN,FN,FP,TP)
    #
    #
    # # OA=(TP+TN)/(TP+TN+FP+FN)
    # # Kappa=((TP+FN)*(TP+FP)+(FN+TN)*(TN+FP))/(allNum*allNum)
    # # precision=TP/(TP+FP)
    # # recall=TP/(TP+FN)
    # # F1=2*precision*recall/(precision+recall)
    # # print(OA,Kappa,precision,recall,F1)
    # # return [OA,Kappa,precision,recall,F1]
    # return [TN,FP,FN,TP]

def sbatch_erroer_matrix(basevenu):
    '''
    根据河段数验证精度
    :param basevenu:
    :return:
    '''
    thresholds = range(450,451,50)

    streamVenu = os.path.join(basevenu, 'Stream3')  # 存放梯度河网阈值的结果
    TstreamFile = os.path.join(basevenu,"visual_stream.tif")    # "visual_stream.tif": NHD  # visual_stream.tif
    # outJson = os.path.join(basevenu, "s_valid_visual_3_25519.csv")
    outJson1 = os.path.join(basevenu,"c_valid_visual_number.csv")  #    _OSM    _1

    single = []
    combina = []
    for threshold in thresholds:

        Venu = os.path.join(streamVenu, str(threshold))
        if not os.path.exists(Venu):
            continue
        slinkFile = os.path.join(Venu, 'slink.tif')

        temp_result = error_matrix(TstreamFile,slinkFile,slinkFile)
        single.append([threshold]+temp_result)

        venu = os.path.join(Venu,"venu")
        for k in range(-60,100,1):

            combinavenu = os.path.join(venu,str(k))
            if not os.path.exists(combinavenu):
                continue
            c_slink = os.path.join(combinavenu,"modified_link.tif")
            temp_result2 = error_matrix(TstreamFile,c_slink,slinkFile)
            combina.append([threshold,k] + temp_result2)
        print("{:s}计算完成".format(Venu))

    # with open(outJson,'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(single)
    #     f.close()
    with open(outJson1, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(combina)
        f.close()

def error_matrix1(trueStreamFile,streamFile):

    """
    将prun后的河网与NHD河网进行匹配，记录重叠的栅格数.

    :param trueStreamFile:
    :param streamFile:
    :param outFile:
    :return:
    """

    Tstream = Raster.get_raster(trueStreamFile)
    stream = Raster.get_raster(streamFile)
    proj,geo,tnodata = Raster.get_proj_geo_nodata(trueStreamFile)
    _,_,s_nodata = Raster.get_proj_geo_nodata(streamFile)
    row,col = Tstream.shape

    TNum1 = 0   # refind stream的数量
    TNum2 = 0   # NHD stream 的数量
    allNum = 0  # 所有的数量
    TP = 0      # 正确的数量：二者重叠
    for i in range(row):
        for j in range(col):
            if Tstream[i,j] != tnodata or stream[i,j] != s_nodata:
                allNum += 1


            if Tstream[i,j] != tnodata and stream[i,j] != s_nodata:
                TP += 1

            if stream[i,j] != s_nodata:
                TNum1 += 1

            if Tstream[i,j] != tnodata:
                TNum2 += 1

    return [allNum,TP,TNum1,TNum2]

def sbatch_erroer_matrix1(basevenu):
    '''
    根据河段重叠验证精度
    :param basevenu:
    :return:
    '''
    thresholds = range(450,451,50)

    streamVenu = os.path.join(basevenu, 'Stream3')  # 存放梯度河网阈值的结果
    TstreamFile = os.path.join(basevenu,"visual_stream.tif")  #OSM : "OSM_valid1.tif"   # "visual_stream.tif": NHD  # visual_stream.tif
    # outJson = os.path.join(basevenu, "s_valid_visual_3_25519_intersect.csv")
    outJson1 = os.path.join(basevenu,"c_valid_visual_length.csv")  #    _OSM    _1

    single = []
    combina = []
    for threshold in thresholds:

        Venu = os.path.join(streamVenu, str(threshold))
        if not os.path.exists(Venu):
            continue
        slinkFile = os.path.join(Venu, 'slink.tif')

        temp_result = error_matrix(TstreamFile,slinkFile,slinkFile)
        single.append([threshold]+temp_result)

        venu = os.path.join(Venu,"venu")
        for k in range(-60,100,1):

            combinavenu = os.path.join(venu,str(k))
            if not os.path.exists(combinavenu):
                continue
            c_slink = os.path.join(combinavenu,"modified_link.tif")
            temp_result2 = error_matrix1(TstreamFile,c_slink)
            combina.append([threshold,k] + temp_result2)
        print("{:s}计算完成".format(Venu))

    # with open(outJson,'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(single)
    #     f.close()
    with open(outJson1, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(combina)
        f.close()



# def reference_stream(rasterStremfile,basestreamfile):
#     """
#     匹配矢量栅格化河网
#     :param rasterStremfile:
#     :param basestreamfile:
#     :return:
#     """
#
#     baseStream = Raster.get_raster(basestreamfile)
#     proj,geo,b_nodata = Raster.get_proj_geo_nodata(basestreamfile)
#
#     rasterStrem = Raster.get_raster(rasterStremfile)
#     proj,geo,r_nodata = Raster.get_proj_geo_nodata(rasterStremfile)
#
#     row,col = baseStream.shape
#
#     result = np.zeros((row,col),dtype = np.int8)
#     for i in range(row):
#         for j in range(col):
#
#             if baseStream[i,j] != b_nodata and rasterStrem[i,j] != r_nodata:
#                 result[i,j] = 1
#
#     Raster.save_raster(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\Wabash River Lower Basin\region3\rundata\visual_stream.tif',result,proj,geo,gdal.GDT_Byte,0)


