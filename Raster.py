from osgeo import gdal,ogr

def GetExtent(in_fn):
    """
    获取影像的最小大坐标
    :param in_fn:
    :return:
    """
    ds=gdal.Open(in_fn)
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    ds=None
    return min_x,max_y,max_x,min_y

def get_raster(file):
    """
    Open raster file and get the raster band
    :param file: the path of file
    :return: array
    """
    datasource=gdal.Open(file)
    raster=datasource.GetRasterBand(1)
    raster=raster.ReadAsArray()
    datasource=None
    return raster
def get_proj_geo_nodata(file):
    """
    Open raster file and get the projection/Geo_Trans/NodataValue of the raster
    :param file: the path of raster
    :return:[proj,geo_trans,nodata]
    """
    datasource=gdal.Open(file)
    raster=datasource.GetRasterBand(1)
    nodata=raster.GetNoDataValue()
    proj=datasource.GetProjection()
    geo_trans=datasource.GetGeoTransform()
    datasource=None
    return [proj,geo_trans,nodata]

def save_raster(outfile,band,proj,geo_trans,datatype,nodata=255):
    row,col=band.shape
    driver=gdal.GetDriverByName("GTIFF").Create(outfile,col,row,1,datatype)
    driver.SetProjection(proj)
    driver.SetGeoTransform(geo_trans)
    # print(1)
    outband=driver.GetRasterBand(1)
    outband.SetNoDataValue(nodata)
    outband.WriteArray(band)
    outband.SetNoDataValue(nodata)
    del driver
    print("{:s}存储成功".format(outfile))

