# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/13 15:15
@File ：Draw.py
@IDE ：PyCharm
"""
import math
import os
import seaborn as sns
import pandas as pd
from osgeo import gdal,ogr
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import json
import csv
import rasterio
import Raster
from Judge_by_Surface_Morphology import *
import os
import geopandas as gpd


def color(r,g,b):
    return (r/255,g/255,b/255)

def Draw_scatter(subbemdding_json):

    with open(subbemdding_json, 'r', encoding='UTF-8') as f:
        result = json.load(f)
    data=[]
    dH=[]
    xy=[]
    for i in result:
        temp_xy=[i]
        for c in result[i]:
            temp_xy.append(result[i][c])
        xy.append(temp_xy)
        if -1<result[i]['subbemdding']<=1 :
            data.append(result[i]['subbemdding'])
            dH.append(result[i]['dH'])
    print(xy)
    with open(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\NWEI\沱沱河\1\基础数据\sub.csv',"a", encoding='utf-8', newline='') as f:
        writer=csv.writer(f)
        writer.writerows(xy)
    # x=[i for i in range(len(data))]
    # plt.scatter(x,data)
    # plt.show()

def draw_F1(csv_file):
    """
    绘制F1变化图
    :param csv_file:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    embedding = []
    F1 = []
    Kappa = []
    accuracy = []
    precision = []
    recall = []
    n=0
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        for i in reader:
            if n>0:
                accuracy.append(float(i[1]))
                precision.append(float(i[3]))
                recall.append(float(i[4]))
                embedding.append(float(i[0]))
                Kappa.append(float(i[2]))
                F1.append(float(i[5]))
            n+=1

    plt.plot(embedding,F1)
    plt.plot([0.453,0.453],[0,1],linestyle='--')
    plt.plot([-0.4,0.6], [0.91,0.91], linestyle='--')
    plt.fill_between(embedding,F1,color='blue',alpha=0.1)
    # plt.fill_between(embedding, accuracy,  alpha=0.1)
    # plt.fill_between(embedding, precision,  alpha=0.1)
    # plt.fill_between(embedding, recall,  alpha=0.1)
    # plt.fill_between(embedding, Kappa,  alpha=0.1)
    # plt.text(0.15,0.2,'embedding=0.453',size=10)
    # plt.xlabel('embedding',size=12)
    # plt.ylabel('ratio',size=12)

    # 0.453
    # 0.829360659 Accuracy
    # 0.827316903 Kappa
    # 0.840362812 Precision
    # 0.983545648 Recall
    # 0.906334067 F1

    plt.show()

def draw_verification(csv_file):
    """
    绘制F1变化图
    :param csv_file:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    embedding = []
    F1 = []
    Kappa = []
    accuracy = []
    precision = []
    recall = []
    n=0
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        for i in reader:
            if n>0:
                accuracy.append(float(i[1]))
                precision.append(float(i[3]))
                recall.append(float(i[4]))
                embedding.append(float(i[0]))
                Kappa.append(float(i[2]))
                F1.append(float(i[5]))
            n+=1


    plt.plot([0.453,0.453],[0,1.05],linestyle='--')
    # plt.plot([-0.4,0.6], [0.91,0.91], linestyle='--')


    plt.subplot(2,2,1)
    plt.plot(embedding, precision,color=color(154,201,219))
    plt.fill_between(embedding, precision,color=color(154,201,219),  alpha=0.4)
    plt.subplot(2,2, 2)
    plt.plot(embedding, recall,color=color(248,172,140))
    plt.fill_between(embedding, recall,color=color(248,172,140),  alpha=0.1)
    plt.subplot(2,2, 3)
    plt.plot(embedding, Kappa,color=color(200,36,35))
    plt.fill_between(embedding, Kappa,color=color(200,36,35),  alpha=0.1)
    plt.subplot(2,2, 4)
    plt.plot(embedding, accuracy, color=color(40, 120, 181))
    plt.fill_between(embedding, accuracy, color=color(40, 120, 181), alpha=0.1)
    # plt.text(0.15,0.2,'embedding=0.453',size=10)
    # plt.xlabel('embedding',size=12)
    # plt.ylabel('ratio',size=12)


    plt.show()

def draw_error():

    plt.errorbar()

def check_slope(slope_file):

    """
    绘制坡度直方图，看坡度分布模式
    :param slope:
    :return:
    """
    slope = Raster.get_raster(slope_file)
    proj,geo,nodata=Raster.get_proj_geo_nodata(slope_file)

    row,col = slope.shape

    slope_value = []
    for i in range(row):
        for j in range(col):
            if slope[i,j] != nodata:
                slope_value.append(slope[i,j]/math.pi*180)


    print("read successfully!")
    print("Average slope is {:.2}".format(sum(slope_value)/len(slope_value)))
    plt.hist(slope_value,bins=100)
    plt.show()


# 20240821
def draw_3D_surface(DEM_file,outfile=r''):
    # if os.path.exists(outfile):
    #     return

    # Load DEM data
    dem_data = Raster.get_raster(DEM_file)
    dem_data = np.array(dem_data,np.float64)
    min_H = min(dem_data[dem_data>0])
    max_H = max(dem_data[dem_data > 0])
    dem_data[dem_data == max_H] = np.nan
    dem_data[dem_data<=min_H] = np.nan
    try:
        print(min(dem_data[dem_data>0]))
    except:
        return
    # Create grid of coordinates
    x = np.arange(dem_data.shape[1])
    y = np.arange(dem_data.shape[0])
    x, y = np.meshgrid(x, y)
    z = dem_data


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x,y,z,cmap = 'rainbow')
    # ax.contourf(x,y,z,  # 传入数据
    #             zdir='z'  # 设置为z轴为等高线的不变轴
    #             , offset=min_H  # 映射位置在z=-1处
    #             , cmap=plt.get_cmap('rainbow')  # 设置颜色为彩虹色
    #             )  # 绘制图像的映射，就是等高线图。

    # 设置边界的填充颜色和透明度
    surf.set_edgecolors('k')  # 设置边界颜色为黑色
    ax.set_facecolor('white')  # 设置图形背景颜色为白色/
    # 调整视角
    ax.view_init(elev=18, azim=140)
    ax.axis('off')

    # plt.savefig(outfile)
    # plt.close()
    plt.show()
def draw_3D_surface_longest_stream(DEM_file,stream_file,outfile=r''):
    # if os.path.exists(outfile):
    #     return

    # Load DEM data
    dem_data = Raster.get_raster(DEM_file)
    dem_data = np.array(dem_data,np.float64)
    min_H = min(dem_data[dem_data>0])
    max_H = max(dem_data[dem_data > 0])
    dem_data[dem_data == max_H] = np.nan
    dem_data[dem_data<=min_H] = np.nan


    s_data = Raster.get_raster(stream_file)
    s_data = np.array(s_data,np.float64)
    row,col = s_data.shape
    for i in range(row):
        for j in range(col):
            if s_data[i,j] != 1:
                continue
            s_data[i,j] += dem_data[i,j]+1
            for k in range(8):
                next_cell = (i+dmove[k][0],j+dmove[k][1])
                if 0 <= next_cell[0] < row and 0 <= next_cell[1] < col:
                    if dem_data[next_cell[0],next_cell[1]] == np.nan:
                        continue
                    if s_data[next_cell[0],next_cell[1]] == 0:
                        s_data[next_cell[0],next_cell[1]] += dem_data[next_cell[0],next_cell[1]]+1+1

    s_data[s_data==0] = np.nan
    try:
        print(min(dem_data[dem_data>0]))
    except:
        return
    # Create grid of coordinates
    x = np.arange(dem_data.shape[1])
    y = np.arange(dem_data.shape[0])
    x, y = np.meshgrid(x, y)
    z = dem_data


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x,y,z,cmap='rainbow', edgecolor='k',alpha = 0.6)
    # ax.contourf(x,y,z,  # 传入数据
    #             zdir='z'  # 设置为z轴为等高线的不变轴
    #             , offset=min_H  # 映射位置在z=-1处
    #             , cmap=plt.get_cmap('rainbow')  # 设置颜色为彩虹色
    #             )  # 绘制图像的映射，就是等高线图。

    # 设置边界的填充颜色和透明度
    surf.set_edgecolors('k')  # 设置边界颜色为黑色
    ax.set_facecolor('white')  # 设置图形背景颜色为白色/
    # 调整视角
    ax.view_init(elev=18, azim=140)
    ax.axis('off')
    # river_x, river_y = np.where(s_data == 1)  # 找到河网的位置
    # z1 = s_data
    ax.plot_surface(x, y,s_data, color='blue')

    # plt.savefig(outfile)
    # plt.close()
    plt.show()
def Draw_profile():
    #绘制最长流路的坡面
    # 构建研究区最长流路
    stream = Raster.get_raster(
        r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\曲面\raster\delete\s1000link.tif')
    s_proj, s_geo, s_nodata = Raster.get_proj_geo_nodata(
        r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\曲面\raster\delete\s1000link.tif')
    dem = Raster.get_raster(r'E:\察隅野外-202311\察隅采样\察隅流域\DEM\DEM.tif')
    dir = Raster.get_raster(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\Dir.tif')
    _, _, dir_nodata = Raster.get_proj_geo_nodata(
        r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\Dir.tif')
    # A = Stream(stream, dem, dir, s_nodata, dir_nodata)
    # result = A.find_longest_path()
    # print(s_geo)
    id = [5591,3381,3127,5177]
    #   9414    6275  5668   8616
    c = [[(2935, 5423), (2935, 5422), (2935, 5421), (2935, 5420), (2935, 5419), (2935, 5418), (2935, 5417), (2936, 5416), (2937, 5415), (2938, 5415), (2939, 5414), (2940, 5413), (2940, 5412), (2941, 5411), (2941, 5410), (2941, 5409), (2940, 5408), (2940, 5407), (2940, 5406), (2939, 5405), (2939, 5404), (2940, 5403), (2940, 5402), (2941, 5402), (2942, 5402), (2943, 5402), (2944, 5402), (2945, 5402), (2946, 5402), (2947, 5402), (2948, 5403), (2949, 5404), (2950, 5405), (2951, 5406), (2952, 5407), (2953, 5408), (2954, 5409), (2955, 5410), (2956, 5411), (2957, 5412), (2958, 5412), (2959, 5413), (2960, 5414), (2961, 5414), (2962, 5415), (2963, 5416), (2964, 5416), (2965, 5416), (2966, 5416), (2967, 5416), (2968, 5416), (2969, 5416), (2970, 5415), (2971, 5414), (2971, 5413), (2972, 5412), (2973, 5411), (2974, 5411), (2975, 5412), (2976, 5412), (2977, 5412), (2978, 5412), (2979, 5412), (2980, 5412), (2981, 5411), (2982, 5410), (2983, 5410), (2984, 5409), (2985, 5409), (2986, 5408), (2987, 5407), (2988, 5407), (2989, 5407), (2990, 5407), (2991, 5407), (2992, 5407), (2993, 5406), (2994, 5406), (2995, 5406), (2996, 5406), (2997, 5406), (2998, 5406), (2999, 5406), (3000, 5405), (3001, 5405), (3002, 5404), (3003, 5404), (3004, 5403), (3005, 5403), (3006, 5403), (3007, 5402), (3008, 5401), (3009, 5401), (3010, 5401), (3011, 5401)],
    [(1849, 3580), (1848, 3581), (1848, 3582), (1848, 3583), (1849, 3584), (1849, 3585), (1849, 3586), (1850, 3587), (1850, 3588), (1851, 3588), (1852, 3589), (1853, 3589), (1854, 3590), (1855, 3590), (1856, 3590), (1857, 3590), (1858, 3590), (1859, 3590), (1860, 3591), (1861, 3591), (1862, 3591), (1863, 3591), (1864, 3591), (1865, 3591), (1866, 3590), (1867, 3590), (1868, 3590), (1869, 3591), (1870, 3591), (1871, 3591), (1872, 3591), (1873, 3591), (1874, 3591), (1875, 3591), (1876, 3591), (1877, 3591), (1878, 3591), (1879, 3590), (1880, 3589), (1881, 3588), (1881, 3587), (1881, 3586), (1882, 3585), (1883, 3584), (1884, 3583), (1885, 3583), (1886, 3582), (1887, 3581), (1888, 3581), (1889, 3580), (1890, 3579), (1891, 3578), (1892, 3578), (1893, 3578), (1894, 3577), (1895, 3577), (1896, 3577), (1897, 3577), (1898, 3576), (1899, 3575), (1900, 3574), (1901, 3573), (1902, 3573), (1903, 3573), (1904, 3573), (1905, 3573), (1906, 3573), (1907, 3573), (1908, 3572), (1909, 3572), (1910, 3572), (1911, 3572), (1912, 3572), (1913, 3572), (1914, 3572), (1915, 3572), (1916, 3572), (1917, 3572), (1918, 3572), (1919, 3572), (1920, 3573), (1921, 3573), (1922, 3572), (1923, 3572), (1924, 3572), (1925, 3572), (1926, 3572), (1927, 3572), (1928, 3572), (1929, 3571), (1930, 3570), (1931, 3569), (1931, 3568), (1932, 3567), (1932, 3566), (1933, 3565), (1934, 3564), (1935, 3564), (1936, 3563), (1937, 3562), (1937, 3561), (1937, 3560), (1937, 3559), (1937, 3558), (1937, 3557), (1937, 3556), (1937, 3555), (1938, 3554), (1939, 3553), (1940, 3552), (1940, 3551), (1941, 3550), (1942, 3549), (1942, 3548), (1943, 3547), (1944, 3546), (1945, 3545), (1946, 3544), (1947, 3543), (1948, 3542), (1949, 3541), (1950, 3540), (1950, 3539), (1950, 3538), (1951, 3537), (1951, 3536), (1951, 3535), (1951, 3534), (1951, 3533), (1951, 3532), (1951, 3531), (1951, 3530), (1951, 3529), (1951, 3528), (1951, 3527), (1952, 3526), (1952, 3525), (1952, 3524), (1952, 3523), (1952, 3522), (1952, 3521), (1952, 3520), (1953, 3519), (1953, 3518), (1953, 3517), (1953, 3516), (1953, 3515), (1953, 3514), (1954, 3513), (1955, 3512), (1955, 3511), (1955, 3510), (1955, 3509), (1954, 3508), (1955, 3507), (1955, 3506), (1955, 3505), (1955, 3504), (1955, 3503), (1955, 3502)],
    [(1769, 2281), (1770, 2280), (1771, 2279), (1772, 2278), (1773, 2277), (1773, 2276), (1772, 2275), (1771, 2274), (1771, 2273), (1771, 2272), (1771, 2271), (1771, 2270), (1771, 2269), (1772, 2268), (1772, 2267), (1772, 2266), (1772, 2265), (1772, 2264), (1772, 2263), (1772, 2262), (1771, 2261), (1771, 2260), (1771, 2259), (1771, 2258), (1771, 2257), (1771, 2256), (1771, 2255), (1771, 2254), (1771, 2253), (1771, 2252), (1771, 2251), (1771, 2250), (1771, 2249), (1771, 2248), (1771, 2247), (1771, 2246), (1772, 2245), (1773, 2244), (1774, 2243), (1774, 2242), (1774, 2241), (1774, 2240), (1774, 2239), (1774, 2238), (1774, 2237), (1775, 2236), (1776, 2235), (1776, 2234), (1776, 2233), (1776, 2232), (1777, 2231), (1777, 2230), (1777, 2229), (1777, 2228), (1778, 2227), (1779, 2226), (1779, 2225), (1780, 2224), (1781, 2223), (1782, 2222), (1783, 2221), (1784, 2220), (1785, 2219), (1786, 2218), (1787, 2217), (1788, 2216), (1789, 2215), (1790, 2215), (1791, 2214), (1792, 2213), (1792, 2212), (1792, 2211), (1792, 2210), (1792, 2209), (1792, 2208), (1792, 2207), (1792, 2206), (1792, 2205), (1792, 2204), (1793, 2203), (1793, 2202), (1793, 2201), (1794, 2200), (1794, 2199), (1794, 2198), (1794, 2197), (1794, 2196), (1794, 2195), (1793, 2194), (1793, 2193), (1792, 2192), (1791, 2191), (1791, 2190), (1790, 2189), (1790, 2188), (1790, 2187), (1790, 2186), (1790, 2185), (1790, 2184), (1790, 2183), (1790, 2182), (1790, 2181), (1790, 2180), (1790, 2179), (1790, 2178), (1790, 2177), (1790, 2176), (1790, 2175), (1790, 2174), (1790, 2173), (1790, 2172), (1790, 2171), (1790, 2170), (1790, 2169), (1790, 2168), (1789, 2167), (1789, 2166), (1788, 2165), (1788, 2164), (1788, 2163), (1788, 2162), (1788, 2161), (1788, 2160), (1789, 2159), (1789, 2158), (1789, 2157), (1789, 2156), (1790, 2155), (1791, 2154), (1791, 2153), (1792, 2152), (1793, 2151), (1793, 2150), (1794, 2149), (1795, 2148), (1795, 2147), (1796, 2146), (1795, 2145), (1795, 2144), (1795, 2143), (1795, 2142), (1796, 2141), (1797, 2140), (1797, 2139), (1797, 2138), (1797, 2137), (1797, 2136), (1797, 2135), (1797, 2134), (1797, 2133), (1797, 2132), (1798, 2131), (1798, 2130), (1798, 2129), (1798, 2128), (1798, 2127), (1798, 2126), (1798, 2125), (1798, 2124), (1798, 2123), (1798, 2122), (1798, 2121), (1798, 2120), (1798, 2119), (1798, 2118), (1798, 2117), (1798, 2116), (1798, 2115), (1799, 2114), (1799, 2113), (1799, 2112), (1799, 2111), (1799, 2110), (1799, 2109), (1799, 2108), (1799, 2107), (1799, 2106), (1799, 2105), (1799, 2104), (1798, 2103), (1798, 2102)],
    [(2727, 2832), (2726, 2832), (2726, 2831), (2726, 2830), (2726, 2829), (2726, 2828), (2727, 2827), (2728, 2826), (2728, 2825), (2728, 2824), (2727, 2823), (2727, 2822), (2727, 2821), (2727, 2820), (2728, 2819), (2728, 2818), (2728, 2817), (2728, 2816), (2728, 2815), (2728, 2814), (2728, 2813), (2728, 2812), (2727, 2811), (2727, 2810), (2727, 2809), (2727, 2808), (2728, 2807), (2728, 2806), (2729, 2805), (2730, 2804), (2731, 2803), (2731, 2802), (2731, 2801), (2732, 2800), (2732, 2799), (2733, 2798), (2733, 2797), (2733, 2796), (2734, 2795), (2734, 2794), (2734, 2793), (2734, 2792), (2734, 2791), (2734, 2790), (2734, 2789), (2734, 2788), (2735, 2787), (2735, 2786), (2735, 2785), (2735, 2784), (2735, 2783), (2736, 2782), (2736, 2781), (2737, 2780), (2737, 2779), (2737, 2778), (2738, 2777), (2738, 2776), (2738, 2775), (2738, 2774), (2737, 2773), (2737, 2772), (2738, 2771), (2738, 2770), (2738, 2769), (2738, 2768), (2738, 2767), (2739, 2766), (2739, 2765), (2740, 2764), (2741, 2763), (2742, 2762), (2742, 2761), (2742, 2760), (2742, 2759), (2742, 2758)]]
    for sid in range(len(c)):
        # cells = result[sid]
        cells = c[sid]
        print(cells)
        x = [0]
        for k in range(1,len(cells)):
            if abs(cells[k][0]-cells[k-1][0])==0 or abs(cells[k][1]-cells[k-1][1])==0:
                x.append(x[k - 1] + 30 * math.sqrt(2))
            else:
                x.append(x[k - 1] + 30 )


        y = [dem[i[0],i[1]] for i in cells]
        plt.plot(x,y,color='black')
        # plt.axis('equal')    # X Y轴等距
        plt.xlim(min(x)-100, max(x)+100)
        plt.ylim(min(y)-100, max(y)+100)

        # plt.xticks([])
        plt.show()

    # for sid in id:
    #     cells = result[sid]
    #
    #     print(cells)
    #     y = [dem[i[0],i[1]] for i in cells]
    #     plt.plot(y)
    #     plt.show()

def Draw_profile_hillslope():
    # 批量绘制最长流路的坡面
    # 构建研究区最长流路

    # 读取incision数据
    incision_file = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\incision.csv'
    with open(incision_file,'r') as f:
        con = csv.reader(f)

        id_incision = []
        n = 0
        for i in con:
            if n>0:
                id_incision.append((int(i[0]),float(i[1])))
            n+=1
    hillslope_ids = []
    source_watershed_ids = []
    for i in id_incision:
        if i[1]<=0.35:
            hillslope_ids.append(i[0])
        else:
            source_watershed_ids.append(i[0])
    print(len(hillslope_ids))

    stream = Raster.get_raster(
        r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\Stream2000_link.tif')
    s_proj, s_geo, s_nodata = Raster.get_proj_geo_nodata(
        r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\Stream2000_link.tif')
    dem = Raster.get_raster(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\DEM.tif')
    dir = Raster.get_raster(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\Dir.tif')
    _, _, dir_nodata = Raster.get_proj_geo_nodata(
        r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\Dir.tif')
    A = Stream(stream, dem, dir, s_nodata, dir_nodata,r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\Acc2000\Result')
    result = A.find_longest_path()
    # print(s_geo)

    for sid in hillslope_ids:
        if sid in result:
            cells = result[sid]
            # print(cells)
            x = [0]
            for k in range(1, len(cells)):
                if abs(cells[k][0] - cells[k - 1][0]) == 0 or abs(cells[k][1] - cells[k - 1][1]) == 0:
                    x.append(x[k - 1] + 30 * math.sqrt(2))
                else:
                    x.append(x[k - 1] + 30)

            y = [dem[i[0], i[1]] for i in cells]
            plt.plot(x, y, color='black')
            # plt.axis('equal')
            plt.xlim(min(x) - 100, max(x) + 100)
            plt.ylim(min(y) - 100, max(y) + 100)

            # plt.xticks([])
            plt.savefig(os.path.join(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\Hillslope',str(sid)+'.jpg'))
            plt.close()
    # plt.show()

    for sid in source_watershed_ids:
        if sid in result:
            cells = result[sid]

            x = [0]
            for k in range(1, len(cells)):
                if abs(cells[k][0] - cells[k - 1][0]) == 0 or abs(cells[k][1] - cells[k - 1][1]) == 0:
                    x.append(x[k - 1] + 30 * math.sqrt(2))
                else:
                    x.append(x[k - 1] + 30)

            y = [dem[i[0], i[1]] for i in cells]
            plt.plot(x, y, color='black')
            # plt.axis('equal')
            plt.xlim(min(x)-100, max(x)+100)
            plt.ylim(min(y)-100, max(y)+100)
            plt.savefig(
                os.path.join(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\Watershed',
                             str(sid) + '.jpg'))
            plt.close()
            # plt.xticks([])
    # plt.show()

    # ids = [4202]
    # for sid in ids:
    #     if sid in result:
    #         cells = result[sid]
    #
    #         x = [0]
    #         for k in range(1, len(cells)):
    #             if abs(cells[k][0] - cells[k - 1][0]) == 0 or abs(cells[k][1] - cells[k - 1][1]) == 0:
    #                 x.append(x[k - 1] + 30 * math.sqrt(2))
    #             else:
    #                 x.append(x[k - 1] + 30)
    #
    #         y = [dem[i[0], i[1]] for i in cells]
    #         plt.plot(x, y, color='black')
    #         plt.axis('equal')
    #         # plt.xlim(min(x), max(x))
    #         # plt.ylim(min(y), max(y))
    #
    #         # plt.xticks([])
    # plt.show()

def sbatch_get_dem():
    Dem_file = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\DEM.tif'
    watershed_file = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\Watershed.tif'
    venu = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dem'
    if not os.path.exists(venu):
        os.mkdir(venu)
        os.chmod(venu,0o777)
    dem = Raster.get_raster(Dem_file)
    proj,geo,nodata = Raster.get_proj_geo_nodata(Dem_file)
    watershed = Raster.get_raster(watershed_file)
    proj,geo,w_nodata = Raster.get_proj_geo_nodata(watershed_file)

    ids = np.unique(watershed[watershed != w_nodata])
    print(len(ids))
    for new_id in ids:

        # print(new_id)
        mask = np.where(watershed == new_id)
        # print(mask)
        minX = min(mask[0])
        minY = min(mask[1])
        maxX = max(mask[0])
        maxY = max(mask[1])
        # print(minX,minY,maxX,maxY)

        dem_mask = dem[minX:maxX+1,minY:maxY+1].copy()
        watershed_mask = watershed[minX:maxX+1,minY:maxY+1].copy()
        dem_mask[watershed_mask!=new_id] = nodata

        new_geo = (geo[0]+minY*geo[1],geo[1],geo[2],geo[3]+minX*geo[5],geo[4],geo[5])
        dem_path = os.path.join(venu,'dem_'+str(new_id)+'.tif')
        Raster.save_raster(dem_path,dem_mask,proj,new_geo,gdal.GDT_Float32,nodata)
        # break

def sbatch_draw_3D(venu,outvenu):

    files = os.listdir(venu)
    for file in files:
        if file.split('.')[1] !='tif':
            continue
        file_path = os.path.join(venu,file)

        outfile = os.path.join(outvenu,file)
        draw_3D_surface(file_path,outfile)

def get_longest_stream(dem_file,fdir_file,out_file):

    dem = Raster.get_raster(dem_file)
    proj,geo,d_nodata = Raster.get_proj_geo_nodata(dem_file)
    fdir = Raster.get_raster(fdir_file)
    row,col = dem.shape
    flag = False
    for i in range(row):
        for j in range(col):
            now_dir = fdir[i,j]
            if dem[i,j] == d_nodata:
                continue
            next_cell = (i+dmove_dic[now_dir][0],j+dmove_dic[now_dir][1])
            if 0<= next_cell[0] <row and 0<= next_cell[1] <col:
                continue
                # if fdir[next_cell[0],next_cell[1]] == -2147483647:
            flag = True
            Head = (i, j, 0)
            break
        if flag:
            break

    max_cell = [Head[0],Head[1],Head[2]]
    pop_cells = [Head]
    while pop_cells:
        pop_cell = pop_cells.pop()
        up_cells = get_rever_D8(fdir, pop_cell[0], pop_cell[1], 255)
        for up_cell in up_cells:
            pop_cells.append((up_cell[0], up_cell[1], pop_cell[2] + 1))

            if max_cell[2] < pop_cell[2] + 1:
                max_cell = [up_cell[0], up_cell[1], pop_cell[2] + 1]

    longest = np.zeros((row,col))
    pop_cells = [max_cell]
    while pop_cells:
        pop_cell = pop_cells.pop()
        longest[pop_cell[0],pop_cell[1]] = 1
        now_dir = fdir[pop_cell[0],pop_cell[1]]
        if now_dir not in dmove_dic:
            continue
        next_cell = (pop_cell[0] + dmove_dic[now_dir][0], pop_cell[1] + dmove_dic[now_dir][1])
        if 0 <= next_cell[0] < row and 0 <= next_cell[1] < col:
            pop_cells.append(next_cell)

    Raster.save_raster(out_file,longest,proj,geo,gdal.GDT_Byte,0)





def density(Name,basevenu):
    venu = os.path.join(basevenu,'venu')
    # 计算河网密度
    size = 10
    watershed_file = os.path.join(basevenu,"watershed.tif")
    DEM_file = os.path.join(basevenu,"Filleddem.tif")
    watershed = Raster.get_raster(watershed_file)
    proj,geo,w_nodata = Raster.get_proj_geo_nodata(watershed_file)

    # stream = Raster.get_raster(stream_link_file)
    # proj,geo,s_nodata = Raster.get_proj_geo_nodata(stream_link_file)

    DEM = Raster.get_raster(DEM_file)
    print(geo)
    row,col = watershed.shape

    stream_area_dic = {}
    watershed_area = 0

    for i in range(row):
        for j in range(col):
            # if stream[i,j] != s_nodata:
            #     stream_area_dic.setdefault(stream[i,j],[]).append((i,j,DEM[i,j]))
            if watershed[i,j] != w_nodata:
                watershed_area += 1
    watershed_area /= 10000



    x = [0,0.07,0.14,0.21,0.25,0.35]
    y = []
    x1 = []
    y1 = []
    for k in range(0,26,5):
        x1.append(k/100)
        stream_area_dic = {}
        stream_file = os.path.join(venu,str(k),'modified_link.tif')
        Stream = Raster.get_raster(stream_file)
        proj, geo, s_nodata = Raster.get_proj_geo_nodata(stream_file)

        row,col = Stream.shape

        for i in range(row):
            for j in range(col):
                if Stream[i,j] != s_nodata:
                    stream_area_dic.setdefault(Stream[i,j],[]).append((i,j,DEM[i,j]))

        stream_len = 0
        for s_id in stream_area_dic:
            cells = stream_area_dic[s_id].copy()
            cells.sort(key=lambda x: x[2])
            temp_len = 0
            if len(cells) == 1:
                temp_len += size/1000
            else:
                for i in range(len(cells) - 1):
                    start_cell = cells[i]
                    end_cell = cells[i + 1]
                    if (start_cell[0] - end_cell[0] == 0) or (start_cell[1] - end_cell[1] == 0):
                        temp_len += size * math.sqrt(2) / 1000
                    else:
                        temp_len += size / 1000
            stream_len += temp_len
        y.append(stream_len / watershed_area)  # 河网密度
        y1.append(len(stream_area_dic))

    # print(x1,y1)
    # print(x1,y)
    # 绘制河网密度图
    # venu = r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\德克萨斯丘陵'
    density_save_path = os.path.join(basevenu,Name+'_density.svg')
    plt.figure(figsize=(5, 4))
    plt.plot(x1,y,zorder=1)
    plt.scatter(x1, y)
    plt.savefig(density_save_path)
    # plt.show()
    plt.close()

    # 绘制hillslope和watershed的数量
    # number_save_path = os.path.join(venu,Name+'_number.svg')
    # plt.figure(figsize=(5, 4))
    # # plt.scatter(x, x1,color=color(250,127,111))
    # # plt.plot(x,x1,color=color(250,127,111))
    #
    # plt.scatter(x,y1,color=color(130,176,210))
    # plt.plot(x, y1, color=color(130,176,210))
    # plt.savefig(number_save_path)
    # # plt.show()
    # plt.close()

    # # 绘制误差棒
    # errorbar_save_path = os.path.join(venu,Name+'_error.svg')
    # plt.figure(figsize=(5,4))
    # plt.boxplot(incison_list,labels=x,sym='.')
    # plt.xticks(rotation=70,ha='right')
    # # plt.show()
    # plt.savefig(errorbar_save_path)


# ##################### 热图绘制:河网阈值+incision index ##########################
def heapmap_cosis(c_valid_file,outVenu):
    """
    绘制kappa热图
    :param c_valid_file:
    :return:
    """
    # [streamArea, NHDArea, TP, redundant, notcheck]
    data = np.zeros((4000,120))
    minValue = 100
    onotchecks = {}
    with open(c_valid_file,'r') as f:
        reader = csv.reader(f)
        for i in reader:
            # x = int(float(i[0])/50-2)   # mssi:-70
            # x = int(float(i[0])/50-2)   # buffalo_draw_watershed:-60
            # x = int(float(i[0]) / 50 - 2)  # klld: -2
            x = int(float(i[0]) / 50 - 2)  # ablq: -20

            y = int(float(i[1])) + 60
            # [TN, FP, FN, TP]
            StreamArea = int(float(i[2]))
            NHDArea = int(float(i[3]))
            TP = int(float(i[4]))
            redundant = int(float(i[5]))
            notcheck = int(float(i[6]))

            if x not in onotchecks:
                onotchecks[x] = notcheck

            ostreamArea = int(float(i[7]))
            OTPs = int(float(i[8]))
            oredundant = int(float(i[9]))

            if StreamArea == 0 :
                continue


            OA_rate = TP/NHDArea
            redundant_rate = redundant/StreamArea

            notcheck_rate = notcheck/NHDArea
            cosis_rate = 1/(redundant_rate+notcheck_rate)#TP/(redundant + notcheck+TP)
            CSI = TP/(StreamArea+NHDArea - TP)#TP / (redundant + notcheck + TP)

            # # 混淆矩阵
            # confusion_matrix = np.array([[0, redundant_rate], [notcheck_rate, OA_rate]])
            # # 总样本数
            # total = np.sum(confusion_matrix)
            # # 观察一致性
            # po = np.trace(confusion_matrix) / total
            # # 每个类别的边际概率
            # row_marginals = np.sum(confusion_matrix, axis=1) / total
            # col_marginals = np.sum(confusion_matrix, axis=0) / total
            # # 预期一致性
            # pe = np.sum(row_marginals * col_marginals)
            # # 计算 Kappa
            # Kappa = (po - pe) / (1 - pe)
            #
            precision = TP/(TP+redundant)
            recall = TP/(TP+notcheck)
            F1 = 2*(precision*recall)/(precision+recall)
            #
            # # 剔除的正确冗余和误判TP的调和F1-score
            # correct = (oredundant - redundant)/(notcheck + 1)
            # incorrect = (OTPs - TP)/(notcheck + 1)
            # F1_scroe = 2*(correct * incorrect) / (1 + correct + incorrect)
            #
            # # 损失比
            M1 = ostreamArea - StreamArea - notcheck + onotchecks[x]
            N1 = M1 + notcheck
            O1 = M1/(1+N1)
            P1 = notcheck/(1+N1)
            loss = O1/(0.0001+P1)

            # 数量CSI
            NumCSI = recall#precision#TP/(NHDArea+redundant+notcheck)


            data[x,y] = NumCSI

            minValue = min(minValue,cosis_rate)
            # print(i)
        f.close()
    # data[data == 0] = minValue
    # 数据标准化到0-1范围
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # normalized_data = scaler.fit_transform(data)
    # normalized_data[data == 0] = np.nan

    # 构造连续的色带 (选择 'viridis' 或其他 colormap)
    # cmap = cm.get_cmap('coolwarm', 40)  # 生成 n_curves 个颜色

    from matplotlib.colors import LinearSegmentedColormap
    # 定义从浅蓝到深蓝的渐变色带
    colors = ["lightblue", "blue"]  # 浅蓝到深蓝
    cmap = LinearSegmentedColormap.from_list("blue_gradient", colors, N=8)
    # 绘制平均趋势图
    from scipy.interpolate import interp1d, CubicSpline
    max_x = []
    max_y = []

    x_low = []  # 第一个高于首数值的incision
    x_high = []  # 最后一个高于首数值的incision
    for k in range(7,8): # 20  albq: 0-8    mssi:10-1900  buffalo_draw_watershed:0-8  klld:5

        # X = range(80)  #   buffalo_draw_watershed:40
        # Y = data[k,20:100]      # buffalo_draw_watershed:20-60

        X = range(100)  # ablq: 55
        Y = data[k, 10:110]  # ablq: 20-75

        # X = range(80)  #  mssi:100
        # Y = data[k, 40:120]  #  mssi:0-100

        # X = range(80)  # klld: 55
        # Y = data[k, 20:100]  # klld: 20-95
        if sum(Y) == 0:
            continue
        X = np.array(X)




        data_array = np.array(Y, dtype='float64')  # 转换为浮点数类型
        # 替换 NaN 为 0
        data_array[np.isnan(data_array)] = 0
        # 转回列表
        converted = data_array.tolist()

        x_new = np.linspace(X.min(), X.max(), 1000)
        # 三次样条插值
        cubic_spline = CubicSpline(X, Y)
        y_spline = cubic_spline(x_new)
        # plt.plot(x_new,y_spline, label=str(500 * (k + 1) * 100 / 1000000), linewidth=1)
        maxvalue = Y.max()
        index = converted.index(maxvalue)
        # if (index-20)/100 in max_y:
        #     continue
        max_x.append((100 + k * 50) * 100 / 1000000)
        max_y.append((index-20)/100)
        # print((index-20)/100)

        firstValue = converted[0]
        for kk in range(0,len(converted)):
            if converted[kk] >= firstValue:
                x_low.append((kk-20)/100)
                break
        for kk in range(len(converted)-1,-1,-1):
            if converted[kk] >= firstValue:
                x_high.append((kk-20)/100)
                break
        print(max_x,x_low)
        # plt.plot(x_new,y_spline, label=str((100 + k * 50) * 100 / 1000000), linewidth=2,color=(54 / 255, (200 - (k) * 10) / 255, 255 / 255))  # ablq:1000
        # plt.plot(x_new,y_spline,label = str((100+k*50)*100/1000000),linewidth = 2 ,color = (54/255,(200-(k)*10)/255,255/255) )   #  buffalo_draw_watershed:3000
        # plt.plot(x_new,y_spline, label=str((100 + k * 50) * 100 / 1000000), linewidth=2,color=(54 / 255, (200 - (k) * 30) / 255, 255 / 255))  # klld:100
        # plt.plot(x_new,y_spline, label=str((100 + k * 50) * 100 / 1000000), linewidth=2,color=(54 / 255, (200 - (k-10) * 1) / 255, 255 / 255))  #   mssi:3500

        # max_x.append((100 + k * 50) * 100 / 1000000)
        # max_y.append((index - 20) / 100)
        # plt.scatter(index, maxvalue, color='r')
        # plt.plot(X,converted,label = str((3000+k*50)*100/1000000),linewidth = 2 ,color = (54/255,(200-(k)*30)/255,255/255) )   #  buffalo_draw_watershed:3000
        # plt.plot(X, converted, label=str((3500 + k * 50) * 100 / 1000000), linewidth=2,color=(54 / 255, (200 - (k) * 3) / 255, 255 / 255))  #   mssi:3500
        # plt.plot(X, converted, label=str((100 + k * 50) * 100 / 1000000), linewidth=2,color=(54 / 255, (200 - (k) * 10) / 255, 255 / 255))  # ablq:1000
        # plt.plot(X, converted, label=str((100 + k * 50) * 100 / 1000000), linewidth=2,color=(54 / 255, (200 - (k-3) * 30) / 255, 255 / 255))  # klld:100
    # 设置x轴刻度标签为字符型

    # buffalo_draw_watershed
    # x_labels = [str((i - 40)/100) for i in range(0, 81, 20)]
    # plt.xticks([i for i in range(0, 81, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置

    # mssi
    # x_labels = [str(i - 60) for i in range(0, 101, 20)]
    # plt.xticks([i for i in range(0, 101, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置

    # ablq
    # x_labels = [str((i - 40)/100) for i in range(0, 81, 20)]
    # plt.xticks([i for i in range(0, 81, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置

    # mssi
    # x_labels = [str((i - 20)/100) for i in range(0, 81, 20)]
    # plt.xticks([i for i in range(0, 81, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置


    # klld
    # x_labels = [str((i - 40)/100) for i in range(0, 81, 20)]
    # plt.xticks([i for i in range(0, 81, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置
    # plt.tick_params(axis='both', which='major', labelsize=12)  # 设置x轴和y轴的主刻度标签大小

    # plt.legend(title=r'$Accmulation\ value\  / \ km^2$',title_fontsize=10,fontsize=10)
    # plt.ylabel('CSI',fontsize = 14)
    # plt.xlabel('incision index',fontsize = 14)
    # plt.show()
    # # plt.savefig(os.path.join(outVenu, "cosis_line.svg"))
    # plt.close()
    # 进行线性拟合
    max_x1 = [(i-50) / 100 for i in range(0, 100, 1)]
    max_y1 = data[7, 10:110]  # ablq: 20-75
    max_x = [(i)/100 for i in range(3, 40, 1)]
    max_y = data[7, 63:100]  # ablq: 20-75
    # 定义一个对数函数形式
    def log_func(x, a, b,d):
        return a * x *x + b*x  +d

        # return a * np.log(b * x + c) + d

    from scipy.optimize import curve_fit
    # # 使用curve_fit拟合数据
    # print(max_x,max_y)
    # params, covariance = curve_fit(log_func,np.array(max_x),np.array(max_y), maxfev=10000)
    # print(params,covariance)
    # # 获取拟合参数
    # a, b ,d= params
    # # print(f"拟合参数: a={a}, b={b}, c={c}, d={d}")
    # #
    # # # 生成拟合曲线
    # y_fit = log_func(np.array(max_x), *params)
    # # 计算残差
    # residuals = np.array(max_y) - y_fit
    # # 计算R²
    # y_mean = np.mean(np.array(max_y))  # 真实值的平均值
    # ss_total = np.sum((np.array(max_y) - y_mean) ** 2)  # 总平方和
    # ss_residual = np.sum((np.array(max_y) - y_fit) ** 2)  # 残差平方和
    # r_squared = 1 - (ss_residual / ss_total)
    # print(r_squared)
    # # 手动计算 MSE
    # mse_manual = np.mean((np.array(max_y) - y_fit) ** 2)
    # print(f'MSE (manual): {mse_manual}')
    #
    # # 手动计算 RMSE
    # rmse_manual = np.sqrt(mse_manual)
    # print(f'RMSE (manual): {rmse_manual}')


    # # ablq
    # plt.text(-0.3,0.23,r'$R^2:{:.3f}$'.format(r_squared))
    # plt.text(-0.3, 0.19, r'$RMSE:{:.3f}$'.format(rmse_manual))
    # plt.text(-0.3, 0.27, r'$y = {:.2f}*x^2+{:.2f}*x+{:.2f},x>0$'.format(a, b,d))

    # buffalo_draw_watershed
    # plt.text(0.015, 0.26, r'$R^2:{:.2f}$'.format(r_squared))
    # plt.text(0.015, 0.25, r'$RMSE:{:.3f}$'.format(rmse_manual))
    # plt.text(0.015, 0.27, r'$y = {:.2f}*x+{:.2f}$'.format(a, b))

    # klld
    # plt.text(0.0125,0.03,r'$R^2:{:.2f}$'.format(r_squared))
    # plt.text(0.0125, 0.01, r'$RMSE:{:.3f}$'.format(rmse_manual))
    # plt.text(0.0125, 0.05, r'$y = {:.2f}*x+{:.2f}$'.format(a,b))

    # mssi
    # plt.text(0.15,0,r'$R^2:{:.2f}$'.format(r_squared))
    # plt.text(0.15, -0.05, r'$RMSE:{:.3f}$'.format(rmse_manual))
    # plt.text(0.15, 0.05, r'$y = {:.2f}*x+{:.2f}$'.format(a, b))

    plt.plot([0, 0], [0.33, 0.5], linestyle='-.', color='grey')
    plt.plot([0.25, 0.25], [0.33, 0.5], linestyle='-.', color='grey')
    plt.plot([-0.3, 0.4], [0.4, 0.4], linestyle='-.', color='grey')
    plt.scatter(max_x1,max_y1,s = 14,edgecolors='black',facecolors='none')
    plt.xticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6])
    plt.ylabel('CSI', fontsize=14)
    plt.xlabel('Incision index', fontsize=14)
    print(max_x,x_low)
    # plt.plot(max_x, y_fit, color='red', label='Fitted line',linestyle = '-')


    # plt.plot(max_x,x_low,linestyle = '-')
    # plt.plot(max_x, x_high, linestyle='-')
    plt.show()

    # # 绘制残差图
    # plt.scatter(max_x, residuals, color='red', label='Residuals')
    # plt.axhline(0, color='black', linestyle='--', label='Zero Line')  # 添加零线（基准线）
    # plt.xlabel('Incision index', fontsize=14)
    # plt.ylabel('Residuals', fontsize=14)
    # x_labels = [str(i / 100) for i in range(0, 41, 10)]
    # plt.xticks([i/100 for i in range(0, 41, 10)], x_labels)  # [1, 2, 3] 对应箱线图的位置
    # # plt.title('Residuals vs. Incision')
    # plt.legend()
    # plt.show()

    # 3. Hartigans' Dip Test

    from scipy.stats import norm
    import diptest
    from diptest import diptest
    def dip_test(data):
        dip, p_value = diptest(data)
        return dip, p_value

    # 4. 检验结果
    single_dip, single_p = dip_test(np.array(max_y))
    # multi_dip, multi_p = dip_test(max_x)

    print("Single Peak Data: Dip Statistic = {:.4f}, p-value = {:.4f}".format(single_dip, single_p))
    # print("Multi Peak Data: Dip Statistic = {:.4f}, p-value = {:.4f}".format(multi_dip, multi_p))
    # 5. 结果解释
    if single_p > 0.05:
        print("单峰数据: 无法拒绝单峰假设 (p > 0.05)，可能是单峰。")
    else:
        print("单峰数据: 拒绝单峰假设 (p <= 0.05)，可能是多峰。")

def heapmap_cosis_little_yellow_creek_watershed(c_valid_file,intersect_csv_file,outVenu):
    """
    绘制kappa热图
    :param c_valid_file:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # [streamArea, NHDArea, TP, redundant, notcheck]
    data = np.zeros((1,160))
    A = np.zeros((1, 160))
    B = np.zeros((1, 160))
    minValue = 450
    onotchecks = {}
    with open(c_valid_file,'r') as f:
        reader = csv.reader(f)
        for i in reader:

            x = int(float(i[0]) / 50 - 9)  # ablq: -20  -9

            y = int(float(i[1])) + 60
            # [TN, FP, FN, TP]
            StreamArea = int(float(i[2]))
            NHDArea = int(float(i[3]))
            TP = int(float(i[4]))
            redundant = int(float(i[5]))
            notcheck = int(float(i[6]))

            if x not in onotchecks:
                onotchecks[x] = notcheck

            ostreamArea = int(float(i[7]))
            OTPs = int(float(i[8]))
            oredundant = int(float(i[9]))

            if StreamArea == 0 :
                continue


            OA_rate = TP/NHDArea
            redundant_rate = redundant/StreamArea

            notcheck_rate = notcheck/NHDArea
            cosis_rate = 1/(redundant_rate+notcheck_rate)#TP/(redundant + notcheck+TP)
            CSI = TP/(StreamArea+NHDArea - TP)#TP / (redundant + notcheck + TP)

            # # 混淆矩阵
            confusion_matrix = np.array([[0, redundant_rate], [notcheck_rate, OA_rate]])
            # 总样本数
            total = np.sum(confusion_matrix)
            # 观察一致性
            po = np.trace(confusion_matrix) / total
            # 每个类别的边际概率
            row_marginals = np.sum(confusion_matrix, axis=1) / total
            col_marginals = np.sum(confusion_matrix, axis=0) / total
            # 预期一致性
            pe = np.sum(row_marginals * col_marginals)
            # 计算 Kappa
            Kappa = (po - pe) / (1 - pe)

            precision = TP/(TP+redundant)
            recall = TP/(TP+notcheck)
            F1 = 2*(precision*recall)/(precision+recall)

            # 剔除的正确冗余和误判TP的调和F1-score
            correct = (oredundant - redundant)/(notcheck + 1)
            incorrect = (OTPs - TP)/(notcheck + 1)
            F1_scroe = 2*(correct * incorrect) / (1 + correct + incorrect)

            # 损失比
            M1 = ostreamArea - StreamArea - notcheck + onotchecks[x]
            N1 = M1 + notcheck
            O1 = M1/(1+N1)
            P1 = notcheck/(1+N1)
            loss = O1/(0.0001+P1)


            # PR-AUC
            from sklearn.metrics import auc

            # 假设你已有离散的 precision 和 recall 列表（长度相同）

            print(x,y)
            A[x,y] = precision
            B[x,y] = recall

            # 数量CSI
            NumCSI = CSI#redundant#TP#TP/(TP+redundant+notcheck)#TP/(NHDArea+redundant+notcheck)


            data[x,y] = NumCSI

            minValue = min(minValue,cosis_rate)
            # print(i)
        f.close()
    # PR-AUC
    from sklearn.metrics import auc
    precision = np.array(A[0][60:])
    recall = np.array(B[0][60:])
    # print(precision,recall)
    # 1. 按 recall 排序
    idx = np.argsort(recall)
    r = recall[idx]
    p = precision[idx]

    # 2. 用 sklearn 的 auc（对传入的 x=r, y=p 自动做 trapezoidal rule）
    pr_auc = auc(r, p)
    print(f"离散点 PR‑AUC = {pr_auc:.4f}")

    # data[data == 0] = minValue
    # 数据标准化到0-1范围

    # normalized_data = scaler.fit_transform(data)
    # normalized_data[data == 0] = np.nan

    # 构造连续的色带 (选择 'viridis' 或其他 colormap)
    # cmap = cm.get_cmap('coolwarm', 40)  # 生成 n_curves 个颜色

    from matplotlib.colors import LinearSegmentedColormap
    # 定义从浅蓝到深蓝的渐变色带
    colors = ["lightblue", "blue"]  # 浅蓝到深蓝
    cmap = LinearSegmentedColormap.from_list("blue_gradient", colors, N=8)
    # 绘制平均趋势图
    from scipy.interpolate import interp1d, CubicSpline
    max_x = []
    max_y = []

    x_low = []  # 第一个高于首数值的incision
    x_high = []  # 最后一个高于首数值的incision
    for k in range(0,1):

        X = range(160)  # ablq: 55
        Y = data[k, 0:160]  # ablq: 20-75
        if sum(Y) == 0:
            continue
        X = np.array(X)


        data_array = np.array(Y, dtype='float64')  # 转换为浮点数类型
        # 替换 NaN 为 0
        data_array[np.isnan(data_array)] = 0
        # 转回列表
        converted = data_array.tolist()

        x_new = np.linspace(X.min(), X.max(), 1000)
        # 三次样条插值
        cubic_spline = CubicSpline(X, Y)
        y_spline = cubic_spline(x_new)
        # plt.plot(x_new,y_spline, label=str(500 * (k + 1) * 100 / 1000000), linewidth=1)
        maxvalue = Y.max()
        index = converted.index(maxvalue)
        # if (index-20)/100 in max_y:
        #     continue
        max_x.append((100 + k * 50) * 100 / 1000000)
        max_y.append((index-20)/100)
        # print((index-20)/100)

        firstValue = converted[0]
        for kk in range(0,len(converted)):
            if converted[kk] >= firstValue:
                x_low.append((kk-20)/100)
                break
        for kk in range(len(converted)-1,-1,-1):
            if converted[kk] >= firstValue:
                x_high.append((kk-20)/100)
                break
        print(max_x,x_low)
        plt.plot(x_new,y_spline, label='CSI based on the number of stream', linewidth=2,color='red')  # ablq:1000
        # plt.scatter(x_new, y_spline,s = 6,edgecolors='black',facecolors='none')  # 散点图

    # 设置x轴刻度标签为字符型


    lengthX = []
    lengthCSI = []
    with open(intersect_csv_file) as f:
        reader = csv.reader(f)
        for i in reader:
            lengthCSI.append(float(i[3])/float(i[2]))
            lengthX.append((float(i[1]))+60)

        f.close()


    plt.plot(lengthX,lengthCSI,label='CSI based on the length of stream',linewidth=2,color='blue')
    # ablq
    x_labels = [str((i - 60)/100) for i in range(0, 161, 20)]
    plt.xticks([i for i in range(0, 161, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置

    # plt.legend(title=r'$Accmulation\ value\  / \ km^2$',title_fontsize=10,fontsize=10)
    plt.legend(fontsize=14)
    plt.ylabel('CSI',fontsize = 14)
    plt.xlabel('Incision index',fontsize = 14)
    plt.show()
    # plt.savefig(os.path.join(outVenu, "cosis_line.svg"))
    plt.close()
def heapmap_cosis_brown_mountain_watershed(c_valid_file,intersect_csv_file,outVenu):
    """
    绘制kappa热图
    :param c_valid_file:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # [streamArea, NHDArea, TP, redundant, notcheck]
    data = np.zeros((1,120))
    A = np.zeros((1, 120))
    B = np.zeros((1, 120))
    minValue = 450
    onotchecks = {}
    with open(c_valid_file,'r') as f:
        reader = csv.reader(f)
        for i in reader:

            x = int(float(i[0]) / 50 - 9)  # ablq: -20  -9

            y = int(float(i[1])) + 60
            # [TN, FP, FN, TP]
            StreamArea = int(float(i[2]))
            NHDArea = int(float(i[3]))
            TP = int(float(i[4]))
            redundant = int(float(i[5]))
            notcheck = int(float(i[6]))

            if x not in onotchecks:
                onotchecks[x] = notcheck

            ostreamArea = int(float(i[7]))
            OTPs = int(float(i[8]))
            oredundant = int(float(i[9]))

            if StreamArea == 0 :
                continue


            OA_rate = TP/NHDArea
            redundant_rate = redundant/StreamArea

            notcheck_rate = notcheck/NHDArea
            cosis_rate = 1/(redundant_rate+notcheck_rate)#TP/(redundant + notcheck+TP)
            CSI = TP/(StreamArea+NHDArea - TP)#TP / (redundant + notcheck + TP)

            # # 混淆矩阵
            confusion_matrix = np.array([[0, redundant_rate], [notcheck_rate, OA_rate]])
            # 总样本数
            total = np.sum(confusion_matrix)
            # 观察一致性
            po = np.trace(confusion_matrix) / total
            # 每个类别的边际概率
            row_marginals = np.sum(confusion_matrix, axis=1) / total
            col_marginals = np.sum(confusion_matrix, axis=0) / total
            # 预期一致性
            pe = np.sum(row_marginals * col_marginals)
            # 计算 Kappa
            Kappa = (po - pe) / (1 - pe)

            precision = TP/(TP+redundant)
            recall = TP/(TP+notcheck)
            F1 = 2*(precision*recall)/(precision+recall)

            # 剔除的正确冗余和误判TP的调和F1-score
            correct = (oredundant - redundant)/(notcheck + 1)
            incorrect = (OTPs - TP)/(notcheck + 1)
            F1_scroe = 2*(correct * incorrect) / (1 + correct + incorrect)

            # 损失比
            M1 = ostreamArea - StreamArea - notcheck + onotchecks[x]
            N1 = M1 + notcheck
            O1 = M1/(1+N1)
            P1 = notcheck/(1+N1)
            loss = O1/(0.0001+P1)


            # PR-AUC
            from sklearn.metrics import auc

            # 假设你已有离散的 precision 和 recall 列表（长度相同）


            A[x,y] = precision
            B[x,y] = recall

            # 数量CSI
            NumCSI = CSI#redundant#TP#TP/(TP+redundant+notcheck)#TP/(NHDArea+redundant+notcheck)


            data[x,y] = NumCSI

            minValue = min(minValue,cosis_rate)
            # print(i)
        f.close()
    # PR-AUC
    from sklearn.metrics import auc
    precision = np.array(A[0][60:])
    recall = np.array(B[0][60:])
    # print(precision,recall)
    # 1. 按 recall 排序
    idx = np.argsort(recall)
    r = recall[idx]
    p = precision[idx]

    # 2. 用 sklearn 的 auc（对传入的 x=r, y=p 自动做 trapezoidal rule）
    pr_auc = auc(r, p)
    # print(f"离散点 PR‑AUC = {pr_auc:.4f}")

    # data[data == 0] = minValue
    # 数据标准化到0-1范围

    # normalized_data = scaler.fit_transform(data)
    # normalized_data[data == 0] = np.nan

    # 构造连续的色带 (选择 'viridis' 或其他 colormap)
    # cmap = cm.get_cmap('coolwarm', 40)  # 生成 n_curves 个颜色

    from matplotlib.colors import LinearSegmentedColormap
    # 定义从浅蓝到深蓝的渐变色带
    colors = ["lightblue", "blue"]  # 浅蓝到深蓝
    cmap = LinearSegmentedColormap.from_list("blue_gradient", colors, N=8)
    # 绘制平均趋势图
    from scipy.interpolate import interp1d, CubicSpline
    max_x = []
    max_y = []

    x_low = []  # 第一个高于首数值的incision
    x_high = []  # 最后一个高于首数值的incision
    for k in range(0,1):

        X = range(120)  # ablq: 55
        Y = data[k, 0:120]  # ablq: 20-75
        if sum(Y) == 0:
            continue
        X = np.array(X)


        data_array = np.array(Y, dtype='float64')  # 转换为浮点数类型
        # 替换 NaN 为 0
        data_array[np.isnan(data_array)] = 0
        # 转回列表
        converted = data_array.tolist()

        x_new = np.linspace(X.min(), X.max(), 1000)
        # 三次样条插值
        cubic_spline = CubicSpline(X, Y)
        y_spline = cubic_spline(x_new)
        # plt.plot(x_new,y_spline, label=str(500 * (k + 1) * 100 / 1000000), linewidth=1)
        maxvalue = Y.max()
        index = converted.index(maxvalue)
        # if (index-20)/100 in max_y:
        #     continue
        max_x.append((100 + k * 50) * 100 / 1000000)
        max_y.append((index-20)/100)
        # print((index-20)/100)

        firstValue = converted[0]
        for kk in range(0,len(converted)):
            if converted[kk] >= firstValue:
                x_low.append((kk-20)/100)
                break
        for kk in range(len(converted)-1,-1,-1):
            if converted[kk] >= firstValue:
                x_high.append((kk-20)/100)
                break
        # print(max_x,x_low)
        plt.plot(x_new,y_spline, label='CSI based on the number of stream', linewidth=2,color='red')  # ablq:1000
        # plt.scatter(x_new, y_spline,s = 6,edgecolors='black',facecolors='none')  # 散点图

    # 设置x轴刻度标签为字符型


    lengthX = []
    lengthCSI = []
    with open(intersect_csv_file) as f:
        reader = csv.reader(f)
        for i in reader:
            lengthCSI.append(float(i[3])/float(i[2]))
            lengthX.append((float(i[1]))+60)

        f.close()


    plt.plot(lengthX,lengthCSI,label='CSI based on the length of stream',linewidth=2,color='blue')
    # ablq
    x_labels = [str((i - 60)/100) for i in range(0, 121, 20)]
    plt.xticks([i for i in range(0, 121, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置

    # plt.legend(title=r'$Accmulation\ value\  / \ km^2$',title_fontsize=10,fontsize=10)
    plt.legend(fontsize=14)
    plt.ylabel('CSI',fontsize = 14)
    plt.xlabel('Incision index',fontsize = 14)
    plt.show()
    # plt.savefig(os.path.join(outVenu, "cosis_line.svg"))
    plt.close()
def heapmap_cosis_buffalo_draw_watershed(c_valid_file,intersect_csv_file,outVenu):
    """
    绘制kappa热图
    :param c_valid_file:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # [streamArea, NHDArea, TP, redundant, notcheck]
    data = np.zeros((1,120))
    A = np.zeros((1, 120))
    B = np.zeros((1, 120))
    minValue = 450
    onotchecks = {}
    with open(c_valid_file,'r') as f:
        reader = csv.reader(f)
        for i in reader:

            x = int(float(i[0]) / 50 - 9)  # ablq: -20  -9

            y = int(float(i[1])) + 60
            # [TN, FP, FN, TP]
            StreamArea = int(float(i[2]))
            NHDArea = int(float(i[3]))
            TP = int(float(i[4]))
            redundant = int(float(i[5]))
            notcheck = int(float(i[6]))

            if x not in onotchecks:
                onotchecks[x] = notcheck

            ostreamArea = int(float(i[7]))
            OTPs = int(float(i[8]))
            oredundant = int(float(i[9]))

            if StreamArea == 0 :
                continue


            OA_rate = TP/NHDArea
            redundant_rate = redundant/StreamArea

            notcheck_rate = notcheck/NHDArea
            cosis_rate = 1/(redundant_rate+notcheck_rate)#TP/(redundant + notcheck+TP)
            CSI = TP/(StreamArea+NHDArea - TP)#TP / (redundant + notcheck + TP)

            # # 混淆矩阵
            confusion_matrix = np.array([[0, redundant_rate], [notcheck_rate, OA_rate]])
            # 总样本数
            total = np.sum(confusion_matrix)
            # 观察一致性
            po = np.trace(confusion_matrix) / total
            # 每个类别的边际概率
            row_marginals = np.sum(confusion_matrix, axis=1) / total
            col_marginals = np.sum(confusion_matrix, axis=0) / total
            # 预期一致性
            pe = np.sum(row_marginals * col_marginals)
            # 计算 Kappa
            Kappa = (po - pe) / (1 - pe)

            precision = TP/(TP+redundant)
            recall = TP/(TP+notcheck)
            F1 = 2*(precision*recall)/(precision+recall)

            # 剔除的正确冗余和误判TP的调和F1-score
            correct = (oredundant - redundant)/(notcheck + 1)
            incorrect = (OTPs - TP)/(notcheck + 1)
            F1_scroe = 2*(correct * incorrect) / (1 + correct + incorrect)

            # 损失比
            M1 = ostreamArea - StreamArea - notcheck + onotchecks[x]
            N1 = M1 + notcheck
            O1 = M1/(1+N1)
            P1 = notcheck/(1+N1)
            loss = O1/(0.0001+P1)


            # PR-AUC
            from sklearn.metrics import auc

            # 假设你已有离散的 precision 和 recall 列表（长度相同）


            A[x,y] = precision
            B[x,y] = recall

            # 数量CSI
            NumCSI = CSI#redundant#TP#TP/(TP+redundant+notcheck)#TP/(NHDArea+redundant+notcheck)


            data[x,y] = NumCSI

            minValue = min(minValue,cosis_rate)
            # print(i)
        f.close()
    # PR-AUC
    from sklearn.metrics import auc
    precision = np.array(A[0][60:])
    recall = np.array(B[0][60:])
    # print(precision,recall)
    # 1. 按 recall 排序
    idx = np.argsort(recall)
    r = recall[idx]
    p = precision[idx]

    # 2. 用 sklearn 的 auc（对传入的 x=r, y=p 自动做 trapezoidal rule）
    pr_auc = auc(r, p)
    # print(f"离散点 PR‑AUC = {pr_auc:.4f}")

    # data[data == 0] = minValue
    # 数据标准化到0-1范围

    # normalized_data = scaler.fit_transform(data)
    # normalized_data[data == 0] = np.nan

    # 构造连续的色带 (选择 'viridis' 或其他 colormap)
    # cmap = cm.get_cmap('coolwarm', 40)  # 生成 n_curves 个颜色

    from matplotlib.colors import LinearSegmentedColormap
    # 定义从浅蓝到深蓝的渐变色带
    colors = ["lightblue", "blue"]  # 浅蓝到深蓝
    cmap = LinearSegmentedColormap.from_list("blue_gradient", colors, N=8)
    # 绘制平均趋势图
    from scipy.interpolate import interp1d, CubicSpline
    max_x = []
    max_y = []

    x_low = []  # 第一个高于首数值的incision
    x_high = []  # 最后一个高于首数值的incision
    for k in range(0,1):

        X = range(120)  # ablq: 55
        Y = data[k, 0:120]  # ablq: 20-75
        if sum(Y) == 0:
            continue
        X = np.array(X)


        data_array = np.array(Y, dtype='float64')  # 转换为浮点数类型
        # 替换 NaN 为 0
        data_array[np.isnan(data_array)] = 0
        # 转回列表
        converted = data_array.tolist()

        x_new = np.linspace(X.min(), X.max(), 1000)
        # 三次样条插值
        cubic_spline = CubicSpline(X, Y)
        y_spline = cubic_spline(x_new)
        # plt.plot(x_new,y_spline, label=str(500 * (k + 1) * 100 / 1000000), linewidth=1)
        maxvalue = Y.max()
        index = converted.index(maxvalue)
        # if (index-20)/100 in max_y:
        #     continue
        max_x.append((100 + k * 50) * 100 / 1000000)
        max_y.append((index-20)/100)
        # print((index-20)/100)

        firstValue = converted[0]
        for kk in range(0,len(converted)):
            if converted[kk] >= firstValue:
                x_low.append((kk-20)/100)
                break
        for kk in range(len(converted)-1,-1,-1):
            if converted[kk] >= firstValue:
                x_high.append((kk-20)/100)
                break
        # print(max_x,x_low)
        plt.plot(x_new,y_spline, label='CSI based on the number of stream', linewidth=2,color='red')  # ablq:1000
        # plt.scatter(x_new, y_spline,s = 6,edgecolors='black',facecolors='none')  # 散点图

    # 设置x轴刻度标签为字符型


    lengthX = []
    lengthCSI = []
    with open(intersect_csv_file) as f:
        reader = csv.reader(f)
        for i in reader:
            lengthCSI.append(float(i[3])/float(i[2]))
            lengthX.append((float(i[1]))+60)

        f.close()


    plt.plot(lengthX,lengthCSI,label='CSI based on the length of stream',linewidth=2,color='blue')
    # ablq
    x_labels = [str((i - 60)/100) for i in range(0, 121, 20)]
    plt.xticks([i for i in range(0, 121, 20)], x_labels)  # [1, 2, 3] 对应箱线图的位置

    # plt.legend(title=r'$Accmulation\ value\  / \ km^2$',title_fontsize=10,fontsize=10)
    plt.legend(fontsize=14)
    plt.ylabel('CSI',fontsize = 14)
    plt.xlabel('Incision index',fontsize = 14)
    plt.show()
    # plt.savefig(os.path.join(outVenu, "cosis_line.svg"))
    plt.close()


if __name__=='__main__':



    # heapmap_cosis_brown_mountain_watershed(
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\ablq\河流等级\c_valid_visual_2_2556.csv',
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\ablq\河流等级\c_valid_visual_2_2557_intersect.csv',
    #     r'')
    #
    # heapmap_cosis1(
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\Tippecanoe\region2\rundata\c_valid_visual_3_25519.csv',
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\Tippecanoe\region2\rundata\c_valid_visual_3_25519_intersect.csv',
    #     r'')
    # heapmap_cosis_buffalo_draw_watershed(
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\德克萨斯丘陵\new_rundata\c_valid_visual_3_25515.csv',
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\研究区\NHD\德克萨斯丘陵\new_rundata\c_valid_visual_3_25515_intersect.csv',
    #     r'')


    # Supplent 附图
    # 批量绘制剖面高程图
    # Draw_profile_hillslope()
    # sbatch_get_dem()
    # sbatch_draw_3D(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dem',
    #                r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\D_profile')
    # draw_3D_surface(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dem\dem_767.tif')
    # get_longest_stream(r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dem\dem_898.tif',
    #                    r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dir\dir_898.tif',
    #                    r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dir\stream_898.tif')
    # Supplent 附图主代码
    # get_longest_stream(
    # r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\论文\图\草图\valid\3472.tif',
    # r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dir\dir_3472.tif',
    # r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dir\stream_3472.tif')
    #
    # draw_3D_surface_longest_stream(
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\论文\图\草图\valid\3472.tif',
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dir\stream_3472.tif')
    # draw_3D_surface_longest_stream(
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dem\dem_3023.tif',
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dir\stream_3023.tif')
    # draw_3D_surface_longest_stream(
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dem\dem_898.tif',
    #     r'F:\专利申请\一种考虑地表形态特征的子流域与坡面判别方法\DATA\察隅验证\run_data\result\dir\stream_898.tif')



    pass