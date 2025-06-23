# -*- coding: utf-8 -*-
"""
@Time ： 2025/5/6 9:41
@File ：check.py
@IDE ：PyCharm


后续河网分析
"""
import csv
import os.path
import matplotlib.pyplot as plt
import Raster

def check_initial_NHD(NHD_file,stream_file,stream_order_file):

    NHD = Raster.get_raster(NHD_file)
    proj,geo,N_nodata = Raster.get_proj_geo_nodata(NHD_file)
    stream = Raster.get_raster(stream_file)
    proj,geo,s_nodata = Raster.get_proj_geo_nodata(stream_file)
    stream_order = Raster.get_raster(stream_order_file)

    row,col = NHD.shape

    order_dic = {}
    for i in range(row):
        for j in range(col):

            if NHD[i,j] == N_nodata and stream[i,j] != s_nodata:

                order_dic.setdefault(stream[i,j],stream_order[i,j])

    # 统计条数
    order = {}
    for stream_id in order_dic:
        order[order_dic[stream_id]] = order.get(order_dic[stream_id],0) + 1

    print(order)




def check_order(stream_order_file,stream_link_file,streamfile,venu):
    """
     检查每个incision下被剔除的河流等级分布(跟原始河网等级比较)
    :param stream_order_file:
    :param stream_link_file:
    :param streamfile:
    :param venu:
    :return:
    """

    stream_order = Raster.get_raster(stream_order_file)
    result = []
    incisioninsexs = range(-60, 60, 1)
    for incisioninsex in incisioninsexs:
        outvenu = os.path.join(venu, str(incisioninsex))
        try:
            modified_stream_file = os.path.join(outvenu,'modified_stream.tif')
            modified_stream = Raster.get_raster(modified_stream_file)
            proj,geo,ms_nodata = Raster.get_proj_geo_nodata(modified_stream_file)
            modified_stream[modified_stream == ms_nodata] = 0

            stream = Raster.get_raster(streamfile)
            proj,geo,s_nodata = Raster.get_proj_geo_nodata(streamfile)
            stream[stream == s_nodata] = 0

            stream_link = Raster.get_raster(stream_link_file)

            mask = stream - modified_stream

            row,col = mask.shape

            stream_order_dic = {}
            for i in range(row):
                for j in range(col):
                    if mask[i,j] == 0:
                        continue
                    stream_order_dic.setdefault(stream_order[i,j],set()).add(stream_link[i,j])   # 存储每个等级对应的河流id，即为了后续计算每个等级下被剔除的河流数量


            temp = [incisioninsex]
            for k in range(1,11):

                if k not in stream_order_dic:
                    temp.append(0)
                else:
                    temp.append(len(stream_order_dic[k]))
            result.append(temp)

        except Exception as e:
            print(incisioninsex, ':', e)


    return result

def check_order_main(basevenu):
    """
    检查每个incision下被剔除的河流等级分布(跟原始河网等级比较)，生成csv文件，供后续作图
    :param baseVenu:
    :return:
    """

    thresholds = range(450, 451, 50)

    stream_oreder_file = os.path.join(basevenu, 'stream_order.tif')
    streamVenu = os.path.join(basevenu, 'Stream')  # 存放梯度河网阈值的结果


    for threshold in thresholds:

        Venu = os.path.join(streamVenu, str(threshold))
        stream_file = os.path.join(Venu, 'stream.tif')
        stream_link_file = os.path.join(Venu, 'slink.tif')


        if not os.path.exists(Venu):
            continue

        outvenu = os.path.join(Venu, "venu")
        if not os.path.exists(outvenu):
            continue

        # 核心函数
        result = check_order(stream_oreder_file,stream_link_file,stream_file,outvenu)

        check_file = os.path.join(basevenu,'check.csv')
        with open(check_file,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)
            f.close()

def draw_check_order(check_csv,check_csv2):

    con = []
    with open(check_csv,'r') as f:
        reader = csv.reader(f)
        for i in reader:
            con.append(i)

    con2 = []
    with open(check_csv2, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            con2.append(i)

    for i in range(1,len(con[0])):

        x = []
        y = []

        x1 = []
        y1 = []
        for j in range(len(con)):
            if con[j][i] == '0':
                continue
            x.append(float(con[j][0]))
            y.append(float(con[j][i]))

            if con2[j][i] == '0':
                continue
            x1.append(float(con[j][0]))
            y1.append(float(con2[j][i]))  # 北正确剔除的数量
            # y1.append(float(con2[j][i])/float(con[j][i]))  # 北正确剔除的比例
        if x == []:
            continue
        # plt.plot(x,y,label = str(i))
        plt.plot(x1,y1,label=str(i))
    # plt.title('Number of the pruned stream correctly')
    plt.xlabel('Incision index')
    # plt.ylabel('Number of thr pruned stream segment')
    plt.legend()
    plt.show()

def check_order2(stream_order_file,stream_link_file,streamfile,venu):
    """
     检查每个incision下被剔除的河流等级分布(跟原始河网等级比较)
    :param stream_order_file:
    :param stream_link_file:
    :param streamfile:
    :param venu:
    :return:
    """

    stream_order = Raster.get_raster(stream_order_file)
    result = []
    incisioninsexs = range(-60, 60, 1)
    for incisioninsex in incisioninsexs:
        outvenu = os.path.join(venu, str(incisioninsex))
        try:
            modified_stream_file = os.path.join(outvenu,'modified_stream.tif')
            modified_stream = Raster.get_raster(modified_stream_file)
            proj,geo,ms_nodata = Raster.get_proj_geo_nodata(modified_stream_file)
            modified_stream[modified_stream == ms_nodata] = 0



            stream_link = Raster.get_raster(stream_link_file)
            proj,geo,sl_nodata = Raster.get_proj_geo_nodata(stream_link_file)
            stream = stream_link.copy()
            stream[stream_link == sl_nodata] = 0
            stream[stream_link != sl_nodata] = 1



            mask = stream - modified_stream

            row,col = mask.shape

            stream_order_dic = {}
            for i in range(row):
                for j in range(col):
                    if mask[i,j] <= 0:
                        continue
                    stream_order_dic.setdefault(stream_order[i,j],set()).add(stream_link[i,j])   # 存储每个等级对应的河流id，即为了后续计算每个等级下被剔除的河流数量


            temp = [incisioninsex]
            for k in range(1,11):

                if k not in stream_order_dic:
                    temp.append(0)
                else:
                    temp.append(len(stream_order_dic[k]))
            result.append(temp)

        except Exception as e:
            print(incisioninsex, ':', e)


    return result

def check_order_main2(basevenu):
    """
    检查每个incision下被剔除的河流等级分布(跟NHD河网等级比较)，生成csv文件，供后续作图
    :param baseVenu:
    :return:
    """

    thresholds = range(450, 451, 50)

    stream_oreder_file = os.path.join(basevenu, 'visual_stream_order.tif')
    stream_link_file = os.path.join(basevenu, 'visual_stream.tif')
    streamVenu = os.path.join(basevenu, 'Stream')  # 存放梯度河网阈值的结果


    for threshold in thresholds:

        Venu = os.path.join(streamVenu, str(threshold))
        stream_file = os.path.join(Venu, 'stream.tif')



        if not os.path.exists(Venu):
            continue

        outvenu = os.path.join(Venu, "venu")
        if not os.path.exists(outvenu):
            continue

        # 核心函数
        result = check_order2(stream_oreder_file,stream_link_file,stream_file,outvenu)

        check_file = os.path.join(basevenu,'check2.csv')
        with open(check_file,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)
            f.close()



