import csv
import math
import os
from osgeo import gdal,ogr
import numpy as np
import Raster
import json
import csv
import time
from multiprocessing import Pool


dmove=[(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
dmove_dic = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}

import whitebox
wbt = whitebox.WhiteboxTools()


def get_rever_D8(dir, row, col,Dir_nodata):
    """
    查询输入栅格的上游栅格
    :param dir: array of dir
    :param row: row of the cell
    :param col:
    :return: [(i,j),(),]
    """
    up_cell = []
    row_num, col_num = dir.shape

    for i in range(8):
        now_loc = (row + dmove[i][0], col + dmove[i][1])
        # print(now_loc)
        if 0<=now_loc[0]<row_num and 0<=now_loc[1]<col_num:
            if dir[now_loc[0],now_loc[1]]!=Dir_nodata:
                if dir[now_loc[0], now_loc[1]] == 2 ** ((i + 4) % 8):
                    up_cell.append(now_loc)
    return up_cell

def Check_Head(stream,stream_nodata,dir, row, col):
    """
    判断是否为河流源头
    :param dir: array of dir
    :param row: row of the cell
    :param col:
    :return: [(i,j),(),]
    """
    up_cell = []
    row_num, col_num = dir.shape

    for i in range(8):
        now_loc = (row + dmove[i][0], col + dmove[i][1])
        # print(now_loc)
        if 0<=now_loc[0]<row_num and 0<=now_loc[1]<col_num:
            if dir[now_loc[0], now_loc[1]] == 2 ** ((i + 4) % 8):
                up_cell.append(now_loc)
    for cell in up_cell:
        if stream[cell[0],cell[1]]!=stream_nodata:
            return False

    return True

def Find_outlet(Stream_file,Dir_file,DEM_file):
    Stream=Raster.get_raster(Stream_file)
    Dir=Raster.get_raster(Dir_file)
    DEM=Raster.get_raster(DEM_file)
    proj1,geo1,DEM_nodata=Raster.get_proj_geo_nodata(DEM_file)
    row,col=Dir.shape
    proj,geo,Stream_nodata=Raster.get_proj_geo_nodata(Stream_file)
    _,_,Dir_nodata=Raster.get_proj_geo_nodata(Dir_file)

    # 寻找河流源点
    Head = (-1,-1)
    for i in range(row):
        for j in range(col):
            if Stream[i,j]!=Stream_nodata:
                if Check_Head(Stream,Stream_nodata,Dir,i,j):
                    Head=(i,j)
                    break
                    # return Head

    # 寻找最长上游
    # 使用（loc,length）控制河流长度,存在新数组里，最后找到最大值开始向下游追溯
    upstream_cell_list=[]
    Vis=np.zeros_like(Dir)
    LENGTH=np.zeros_like(Dir)
    if Head==(-1,-1):
        return []
    upstream_cell_pop=[(Head)]
    max_len=0
    max_loc=Head
    while upstream_cell_pop:
        pop_cell=upstream_cell_pop.pop()
        upstreamcell=get_rever_D8(Dir,pop_cell[0],pop_cell[1],Dir_nodata)
        upstream_cell_pop+=upstreamcell
        for cell in upstreamcell:
            LENGTH[cell[0],cell[1]]=LENGTH[pop_cell[0],pop_cell[1]]+1
            if LENGTH[cell[0],cell[1]]>max_len:
                max_len=LENGTH[cell[0],cell[1]]
                max_loc=cell
    # print(LENGTH.max())
    # print(max_loc)

    Head=max_loc
    # 按顺序寻找整条河流并记录坐标
    Stream_cell_list=[max_loc]
    while True:
        # print(Head)
        now_dir=Dir[Head[0],Head[1]]
        if now_dir in dmove_dic:
            next_cell=(Head[0]+dmove_dic[now_dir][0],Head[1]+dmove_dic[now_dir][1])
            if 0<=next_cell[0]<row and 0<=next_cell[1]<col:

                if DEM[next_cell[0],next_cell[1]]!=DEM_nodata:
                    Head = next_cell
                    Stream_cell_list.append(next_cell)
                else:
                    return Stream_cell_list
            else:
                return Stream_cell_list
        else:
            return Stream_cell_list

def Stream_link(Dir_file,Stream_file,Stream_link_file):

    Dir = Raster.get_raster(Dir_file)
    proj,geo,d_nodata = Raster.get_proj_geo_nodata(Dir_file)

    Stream = Raster.get_raster(Stream_file)
    proj,geo,s_nodata = Raster.get_proj_geo_nodata(Stream_file)

    row,col = Dir.shape

    # 寻找1）没有下游的河流点；2）有至少两个上游河流的汇入点
    node = []
    for i in range(row):
        for j in range(col):
            if Stream[i,j] != s_nodata:

                now_dir = Dir[i,j]
                if now_dir not in dmove_dic:
                    continue
                next_cell = (i+dmove_dic[now_dir][0],j+dmove_dic[now_dir][1])
                if 0<= next_cell[0] < row and 0<=next_cell[1]<col:
                    if Dir[next_cell[0],next_cell[1]] == d_nodata:
                        node.append((i,j))


                    up_cells = get_rever_D8(Dir,i,j,d_nodata)
                    if len(up_cells) < 2 :
                        continue
                    up_stream_cell = []
                    for cell in up_cells:
                        if Stream[cell[0],cell[1]] != s_nodata:
                            up_stream_cell.append(cell)
                    if len(up_stream_cell) > 1:
                        # 是
                        node += up_stream_cell

    Stream_link = np.zeros((row,col))
    Stream_link[:,:] = -9999
    stream_id = 1
    for temp_node in node:
        pop_cells = [temp_node]
        while pop_cells:
            pop_cell = pop_cells.pop()
            Stream_link[pop_cell[0],pop_cell[1]] = stream_id
            up_cells = get_rever_D8(Dir,pop_cell[0],pop_cell[1],d_nodata)
            for cell in up_cells:
                if Stream[cell[0],cell[1]] == s_nodata:
                    continue
                if cell not in node:
                    pop_cells.append(cell)
        stream_id+=1
    Raster.save_raster(Stream_link_file,Stream_link,proj,geo,gdal.GDT_Float32,-9999)

class Stream:
    def __init__(self,stream,dem,dir,s_nodata,dir_nodata,venu,watershed_stream_ids=[]):
        self.stream=stream
        self.dem=dem
        self.dir=dir
        self.s_nodata=s_nodata
        self.dir_nodata=dir_nodata
        # self.TWI=TWI
        # self.Convergence=Convergence
        self.row=dir.shape[0]
        self.col=dir.shape[1]
        self.dmove=[(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        self.dmove_dic = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
        self.venu = venu
        self.watershed_stream_ids = watershed_stream_ids

    def dic2csv(self,dic):
        if not os.path.exists(self.venu):
            os.mkdir(self.venu)
        outfile = os.path.join(self.venu,'incision.csv')
        con = [['FID', 'embedding']]
        for id in dic:
            temp_con = [id, dic[id]]
            con.append(temp_con)

        with open(outfile, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(con)
        f.close()

    def find_stream(self):
        result={}

        for i in range(self.row):
            for j in range(self.col):
                if self.stream[i,j]!=self.s_nodata:
                    result.setdefault(self.stream[i,j],[]).append((i,j,self.dem[i,j]))
        return result

    def sort_by_elv(self):
        start_time=time.time()
        result=self.find_stream()
        for stream_id in result:
            temp_stream=result[stream_id]
            result[stream_id]=sorted(temp_stream,key=lambda x:x[2],reverse=True)
        end_time = time.time()
        print("Find stream successfully! Time consume {:.2f}s".format(end_time-start_time))
        return result

    def find_start(self):

        LENGTH = np.zeros_like(self.dir)
        stream=self.sort_by_elv()
        start_time = time.time()
        result={}
        for stream_id in stream:
            Head=stream[stream_id][0]
            if Head == (-1, -1):
                return []
            upstream_cell_pop = [(Head)]
            max_len = 0
            max_loc = Head
            case=1
            while upstream_cell_pop:
                pop_cell = upstream_cell_pop.pop()

                upstreamcell = get_rever_D8(self.dir, pop_cell[0], pop_cell[1], self.dir_nodata)
                # 判断上游是否存在不同的河流id，存在则停止.中间子流域的情况
                flag=0
                for cell in upstreamcell:
                    if self.stream[cell[0],cell[1]]!=self.s_nodata and self.stream[cell[0],cell[1]]!=stream_id:
                        flag=1
                        break
                if flag==1:
                    case=2
                    break
                upstream_cell_pop += upstreamcell
                for cell in upstreamcell:
                    LENGTH[cell[0], cell[1]] = LENGTH[pop_cell[0], pop_cell[1]] + 1
                    if LENGTH[cell[0], cell[1]] > max_len:
                        max_len = LENGTH[cell[0], cell[1]]
                        max_loc = cell
            if case==1 and stream_id not in self.watershed_stream_ids:
                result.setdefault(stream_id,[]).append(max_loc)
        end_time = time.time()
        print("Find ditch sources successfully! Time consume {:.2f}s".format(end_time-start_time))
        return result

    def find_longest_path(self):
        start=self.find_start()
        result={}
        start_time = time.time()
        for stream_id in start:
            Head = start[stream_id][0]
            # 按顺序寻找整条河流并记录坐标
            Stream_cell_list = [Head]
            while True:
                # print(Head)
                now_dir = self.dir[Head[0], Head[1]]
                if now_dir in dmove_dic:
                    next_cell = (Head[0] + dmove_dic[now_dir][0], Head[1] + dmove_dic[now_dir][1])
                    if 0 <= next_cell[0] < self.row and 0 <= next_cell[1] < self.col:
                        if self.dir[next_cell[0], next_cell[1]] != self.dir_nodata:
                            if self.stream[next_cell[0],next_cell[1]] == self.s_nodata:
                                Head = next_cell
                                Stream_cell_list.append(next_cell)
                            elif self.stream[next_cell[0],next_cell[1]] == stream_id:
                                Head = next_cell
                                Stream_cell_list.append(next_cell)
                            else:
                                break
                        else:
                            break
                    else:
                        break
                else:
                    break
            result.setdefault(stream_id,Stream_cell_list)
        end_time = time.time()
        print("Find the longest flow path successfully! Time consume {:.2f}s".format(end_time-start_time))
        return result

    def get_subbemdding(self):
        """
        计算每个subbasin的embedding值，返回list结果，再调用外部函数写入csv
        :return:
        """
        flow_path=self.find_longest_path()
        start_time = time.time()
        result={}
        for stream_id in flow_path:
            Head = flow_path[stream_id][0]
            Tail = flow_path[stream_id][-1]

            max_H = self.dem[Head[0], Head[1]]
            min_H = self.dem[Tail[0], Tail[1]]
            # print(max_H,min_H)
            sum_dh = 0
            X = []
            Y = []
            Z = []
            for cell in flow_path[stream_id]:
                sum_dh += (self.dem[cell[0], cell[1]] - min_H)
                X.append(cell[0])
                Y.append(cell[1])
                Z.append(self.dem[cell[0], cell[1]])

            subembedding = 1 - ((2 * sum_dh) / (len(flow_path[stream_id]) * (max_H - min_H)))
            result.setdefault(stream_id,subembedding)
        end_time = time.time()
        print("Calculate embedding successfully! Time consume {:.2f}s".format(end_time - start_time))
        self.dic2csv(result)

        return result



# 计算不同阈值下的精度
def valid(embedding_file,threshold,valid_file):
    """
    计算在threshold下的各指标结果,valid是验证数据集
    :param embedding_file:
    :param threshold:
    :param valid_file:
    :return:
    """
    hillslope=[]
    subwatershed=[]
    with open(embedding_file, 'r') as f:
        csv_reader = csv.reader(f)
        n = 0
        for i in csv_reader:
            if n >= 1:
                if float(i[1])<threshold:
                    hillslope.append(i[0])
                else:
                    subwatershed.append(i[0])
            n += 1
    f.close()
    hillslope_t=[]
    subwatershed_t=[]
    with open(valid_file, 'r') as f:
        csv_reader = csv.reader(f)
        for i in csv_reader:
            if i[1]=='2':
                hillslope_t.append(i[0])
            if i[1]=='3':
                subwatershed_t.append(i[0])
    f.close()
    # print(hillslope_t,hillslope,subwatershed_t,subwatershed)
    h_sum=len(hillslope_t)
    s_sum=len(subwatershed_t)
    h_t=0
    s_t=0
    for i in hillslope:
        if i in hillslope_t:
            h_t+=1
    for i in subwatershed:
        if i in subwatershed_t:
            s_t+=1
    TP=h_t
    FP=len(hillslope)-TP
    TN=s_t
    FN=len(subwatershed)-TN
    OA=(TP+TN)/(TP+TN+FP+FN)
    Kappa=((TP+FN)*(TP+FP)+(FN+TN)*(TN+FP))/(n*n)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*precision*recall/(precision+recall)
    print(len(hillslope),len(subwatershed))
    return [threshold,OA,Kappa,precision,recall,F1]

def type_catchment(csv_file,threshold,watershed_file,output_discriminate_file,watershed_stream_ids=[]):
    """
    根据embedding.csv和watershed.tif判断类型并存储为TIFF文件
    :param csv_file:
    :param threshold:
    :param watershed_file:
    :return:
    """

    watershed=Raster.get_raster(watershed_file)
    proj,geo,nodata=Raster.get_proj_geo_nodata(watershed_file)
    watershed_dic={}
    row, col = watershed.shape
    for i in range(row):
        for j in range(col):
            if watershed[i,j]!=nodata:
                watershed_dic.setdefault(watershed[i,j],[]).append((i,j))

    type_dic={3:[],2:[]}
    n=0
    with open(csv_file,'r') as f:
        reader=csv.reader(f)
        for i in reader:
            if n>0:
                if float(i[1])<threshold:
                    # 坡面
                    type_dic[2]+=watershed_dic[float(i[0])]
                else:
                    type_dic[3]+=watershed_dic[float(i[0])]
                watershed_dic.pop(float(i[0]))
            n += 1
    f.close()
    print("文件读取完毕")

    arr = np.zeros((row, col), dtype=float)
    arr[:, :] = -9999
    for id in watershed_dic:
        for cell in watershed_dic[id]:
            arr[cell[0],cell[1]]=1

    for type_id in type_dic:
        for cell in type_dic[type_id]:
            arr[cell[0], cell[1]] = type_id

    for id in watershed_stream_ids:
        for cell in watershed_dic[id]:
            arr[cell[0],cell[1]]=3

    # os.rmdir(output_discriminate_file)
    if os.path.exists(output_discriminate_file):
        os.remove(output_discriminate_file)
    Raster.save_raster(output_discriminate_file, arr, proj, geo, gdal.GDT_Float32, -9999)

def modify_river(stream_order_file,Dir_file,csv_file,threshold,stream_file,venu,watershed_stream_ids=[]):
    """
    根据判断的结果修正河网
    :param csv_file: 计算的embeddeding
    :param threshold:
    :param stream_file:
    :return:
    """
    modified_stream_file = os.path.join(venu,'modified_stream.tif')
    modified_link_file = os.path.join(venu, 'modified_link.tif')

    # 河网等级约束
    stream_order = Raster.get_raster(stream_order_file)
    proj,geo,so_nodata = Raster.get_proj_geo_nodata(stream_order_file)



    # 根据阈值修正河网，将坡面并到下游流域:只剔除incision<threashold and 原始河网等级较低的 河流 (order <= 2)
    # 修正河网
    stream = Raster.get_raster(stream_file)
    proj, geo, nodata = Raster.get_proj_geo_nodata(stream_file)

    row, col = stream.shape
    stream_loc = {}
    stream_order_dic = {}
    watershed_stream_ids_dic = {}  #存放该id所在的某个像元，便于更新id
    for i in range(row):
        for j in range(col):
            if stream[i, j] != nodata:
                stream_loc.setdefault(stream[i, j], []).append((i, j))
                stream_order_dic.setdefault(stream[i,j],stream_order[i,j])  # 河流id:河流等级
                if stream[i,j] in watershed_stream_ids:
                    watershed_stream_ids_dic.setdefault(stream[i,j],[]).append((i,j))

    n = 0
    watershed_stream = {}
    with open(csv_file,'r') as f:
        reader=csv.reader(f)
        for i in reader:
            if n>0:
                if float(i[1])<threshold and stream_order_dic[float(int(i[0]))] <= 3:  #  and stream_order_dic[i[0]] <= 2
                    # 坡面河流
                    stream_loc[float(i[0])]=[]
                else:
                    watershed_stream[float(i[0])]=stream_loc[float(i[0])].copy()
            n += 1
    f.close()
    print("文件读取完毕")

    modified_stream=np.zeros((row,col))
    modified_stream[:,:]=-9999


    for s_id in stream_loc:
        for cell in stream_loc[s_id]:
            modified_stream[cell[0],cell[1]]=1
    if os.path.exists(modified_stream_file):
        os.remove(modified_stream_file)
    Raster.save_raster(modified_stream_file,modified_stream,proj,geo,gdal.GDT_Float32,-9999)
    modified_stream_file_vector = os.path.join(venu,'modified_stream.shp')
    wbt.raster_streams_to_vector(modified_stream_file,Dir_file,modified_stream_file_vector,True)

    Stream_link(Dir_file,modified_stream_file,modified_link_file)

    modified_link = Raster.get_raster(modified_link_file)
    proj,geo,l_nodata = Raster.get_proj_geo_nodata(modified_link_file)
    max_id = modified_link[modified_link!=l_nodata].max()
    watershed_stream_ids = []

    # 更新判断为流域的河流
    for id in watershed_stream:
        for cell in watershed_stream[id]:
            modified_link[cell[0],cell[1]] = max_id+1
            watershed_stream_ids.append(max_id+1)
        max_id += 1

    # 记录原来被判断为流域的id
    for id in watershed_stream_ids_dic:
        cells = watershed_stream_ids_dic[id]
        for cell in cells:
            modified_link[cell[0], cell[1]] = max_id + 1
            watershed_stream_ids.append(max_id + 1)
        # watershed_stream_ids.append(modified_link[cell[0],cell[1]])
        max_id+=1
    if os.path.exists(modified_link_file):
        os.remove(modified_link_file)
    Raster.save_raster(modified_link_file,modified_link,proj,geo,gdal.GDT_Float32,l_nodata)

    return watershed_stream_ids
def cal_watershed_area(watershed1_file,watershed2_file):

    watershed1=Raster.get_raster(watershed1_file)
    watershed2=Raster.get_raster(watershed2_file)
    proj,geo,nodata=Raster.get_proj_geo_nodata(watershed1_file)
    # 计算每个流域的平均面积和标准差
    def avg_area(watershed):
        dic = {}
        row,col=watershed.shape
        for i in range(row):
            for j in range(col):
                if watershed[i,j] != nodata:
                    dic.setdefault(watershed[i,j],[]).append((i,j))
        area = [len(dic[i])*33.05042427*33.05042427/1000000 for i in dic]
        avg,std,var=np.average(area),np.std(area),np.var(area)
        return avg,std,var

    print(avg_area(watershed1))
    print(avg_area(watershed2))

def get_basin_embedding(stream_order_file,DEM_file,Dir_file,Stream_file,watershed_file,venu,incision_threshold=0.25):
    if not os.path.exists(venu):
        os.mkdir(venu)
        os.chmod(venu,0o777)

    # 计算embedding的主程序
    stream=Raster.get_raster(Stream_file)
    s_proj,s_geo,s_nodata=Raster.get_proj_geo_nodata(Stream_file)
    dem=Raster.get_raster(DEM_file)
    dir=Raster.get_raster(Dir_file)
    _,_,dir_nodata=Raster.get_proj_geo_nodata(Dir_file)

    # 1、计算类Stream
    A=Stream(stream,dem,dir,s_nodata,dir_nodata,venu)
    # B=A.get_subbemdding()
    # if not os.path.exists(venu):
    #     os.mkdir(venu)
    incision_file = os.path.join(venu,'incision.csv')
    # # A.dic2csv(B,incision_file)
    # print(B)


    # # 2、根据生成的embedding.csv判断subbasin的类型
    output_discriminate_file = os.path.join(venu,'Discriminate.tif')
    # type_catchment(incision_file,incision_threshold,watershed_file,output_discriminate_file)

    # 26721 13230 237
    # 3、根据判断的结果修正河网
    # modify_river(Dir_file,incision_file,incision_threshold,Stream_file,venu)   # 0.25


    # 迭代
    watershed_stream_ids = []
    for _ in range(100):
        print('*******************************************',watershed_stream_ids)
        stream_intial = stream.copy()
        stream_intial[stream_intial!=s_nodata] = 1
        A=Stream(stream,dem,dir,s_nodata,dir_nodata,venu,watershed_stream_ids)
        B=A.get_subbemdding()
        if not os.path.exists(venu):
            os.mkdir(venu)
        incision_file = os.path.join(venu,'incision.csv')
        watershed_file = os.path.join(venu,'watershed.tif')
        wbt.watershed(Dir_file,Stream_file,watershed_file,True)   # 计算流域


        # 2、根据生成的embedding.csv判断subbasin的类型
        output_discriminate_file = os.path.join(venu,'Discriminate.tif')
        type_catchment(incision_file,incision_threshold,watershed_file,output_discriminate_file,watershed_stream_ids)

        # 26721 13230 237
        # 3、根据判断的结果修正河网
        watershed_stream_ids = modify_river(stream_order_file,Dir_file,incision_file,incision_threshold,Stream_file,venu,watershed_stream_ids)   # 0.25
        modified_link_file = os.path.join(venu, 'modified_link.tif')
        stream = Raster.get_raster(modified_link_file)
        s_proj, s_geo, s_nodata = Raster.get_proj_geo_nodata(modified_link_file)
        Stream_file = modified_link_file

        stream_end = stream.copy()
        stream_end[stream_end!=s_nodata] = 1

        if np.all(stream_intial == stream_end):
            print(_)
            break


def sbatch_get_basin_embedding(stream_order_file,DEM_file,Dir_file,Stream_file,watershed_file,venu):
    """
    计算单个河网阈值下的梯度incision index的结果
    :param DEM_file:
    :param Dir_file:
    :param Stream_file:
    :param watershed_file:
    :param venu:
    :return:
    """

    if not os.path.exists(venu):
        os.mkdir(venu)
        os.chmod(venu,0o777)
    incisioninsexs = range(-60,100,1)
    for incisioninsex in incisioninsexs:
        outvenu = os.path.join(venu,str(incisioninsex))
        try:
            get_basin_embedding(stream_order_file,DEM_file, Dir_file, Stream_file, watershed_file, outvenu, incisioninsex/100)
        except Exception as e:
            print(incisioninsex,':',e)


def sbatch_get_basin_embedding_combination(basevenu):
    """
    批量计算组合阈值下的结果
    :param basevenu:
    :return:
    """

    # 回调函数
    def callback(result):
        print(f"Callback received result: {result}")

    thresholds = range(450,451,50)

    dirFile = os.path.join(basevenu, 'dir.tif')
    demFile = os.path.join(basevenu,'Filleddem.tif')
    stream_oreder_file = os.path.join(basevenu,'stream_order.tif')

    streamVenu = os.path.join(basevenu, 'Stream3')  # 存放梯度河网阈值的结果

    params = []
    for threshold in thresholds:

        Venu = os.path.join(streamVenu, str(threshold))
        if not os.path.exists(Venu):
            continue
        slinkFile = os.path.join(Venu, 'slink.tif')
        watershedFile = os.path.join(Venu, 'watershed.tif')

        outvenu = os.path.join(Venu,"venu")
        if not os.path.exists(outvenu):
            os.mkdir(outvenu)
            os.chmod(outvenu,0o777)
        params.append([stream_oreder_file,demFile,dirFile,slinkFile,watershedFile,outvenu])
        # sbatch_get_basin_embedding(demFile,dirFile,slinkFile,watershedFile,outvenu)

        print("{:s}计算完成".format(Venu))
    # po = Pool(3)
    for param in params:
        # po.apply_async(sbatch_get_basin_embedding,(param[0],param[1],param[2],param[3],param[4],param[5],),callback=callback)
        sbatch_get_basin_embedding(param[0], param[1], param[2], param[3], param[4], param[5])
    # po.close()
    # po.join()







