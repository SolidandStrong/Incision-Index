Code Description
Python libraries required to run this project:
Osgeo, csv, matplotlib, math, os, numpy, json, time, multiprocessing, seaborn, pandas, rasterio, geopandas, whitebox

Organize each watershed in the given form before running, see the code for details. Unzip the contents of DATA.zip to the code directory in advance.

1. Run the main function
The main processing function is stored in Judge_by_Surface_Morphology.py
1.1 Produce a river network, run single_acc_threshold.sbatch_extract_stream, and modify the threshold at line 32 of single_acc_threshold.py according to user needs
1.2 Enter the main processing stream and run Judge_by_Surface_Morphology.sbatch_get_basin_embedding_combination
1.3 Verify the CSI based on the number of river sections, run valid.sbatch_erroer_matrix
1.4 Verify the CSI based on the length of the river section, run valid.sbatch_erroer_matrix1

2. Drawing function
The drawing function is stored in Draw.py
2.1 Figure1: Drawing with Visio and PPT, no code
2.2 Figure2: Drawing with PPT, no code
2.3 Figure3: Drawing with ArcGIS software, no code
2.4 Figure 4: Draw with ArcGIS software, no code. The data is the result of step 1
2.5 Figure 5: Run Draw.heapmap_cosis_brown_mountain_watershed, Draw.heapmap_cosis_little_yellow_creek_watershed, Draw.heapmap_cosis_buffalo_draw_watershed, the data is the result of 1.3 and 1.4.
 
代码说明
运行该项目所需的Python库：
Osgeo、csv、matplotlib、math、os、numpy、json、time、multiprocessing、seaborn、pandas、rasterio、geopandas、whitebox

运行前将每个流域按照给定的形式进行组织，详情参见代码。提前将DATA.zip的内容解压到代码目录。

1、	运行主函数
主要处理函数存放在Judge_by_Surface_Morphology.py
1.1	生产河网，运行single_acc_threshold.sbatch_extract_stream，根据用户需求single_acc_threshold.py的32行处修改阈值
1.2	进入主要处理流，运行Judge_by_Surface_Morphology.sbatch_get_basin_embedding_combination
1.3	验证基于河段数量的CSI，运行valid.sbatch_erroer_matrix
1.4	验证基于河段长度的CSI，运行valid.sbatch_erroer_matrix1

2、	绘图函数
绘图函数存放在Draw.py
2.1 Figure1：使用Visio和PPT进行绘制，无代码
2.2 Figure2：使用PPT进行绘制，无代码
2.3 Figure3：使用Arcgis软件绘制，无代码
2.4 Figure4：使用Arcgis软件绘制，无代码。数据为步骤1的结果
2.5 Figure5：运行Draw.heapmap_cosis_brown_mountain_watershed, Draw.heapmap_cosis_little_yellow_creek_watershed, Draw.heapmap_cosis_buffalo_draw_watershed，数据为1.3和1.4的结果。

