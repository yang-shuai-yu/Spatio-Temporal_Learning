import os, pickle
import numpy as np
import shapely.geometry as sg
from shapely.geometry import Point
import math
import pandas as pd

EARTH_RADIUS = 6381372    # meter 

#定义经纬度转换为米勒坐标的方法
def millerToXY(lon, lat):
    xy_coordinate = []
    L = EARTH_RADIUS * math.pi * 2  #地球周长
    W = L  #平面展开，将周长视为X轴
    H = L / 2  #Y轴约等于周长一半
    mill = 2.3  #米勒投影中的一个常数，范围大约在正负2.3之间
    #循环，因为要批量转换
    for x, y in zip(lon, lat):
        x = x * np.pi / 180  # 将经度从度数转换为弧度
        y = y * np.pi / 180  # 将纬度从度数转换为弧度
        y = 1.25 * np.log(np.tan(0.25 * np.pi + 0.4 * y))  # #这里是米勒投影的转换
        x = (W / 2) + (W / (2 * np.pi)) * x  #这里将弧度转为实际距离 ，转换结果的单位是km
        y = (H / 2) - (H / (2 * mill)) * y  # 这里将弧度转为实际距离 ，转换结果的单位是km
        xy_coordinate.append([np.around(x).astype(int), np.around(y).astype(int)])
    xy_coordinate = np.array(xy_coordinate)
    return xy_coordinate

#xy坐标转换成经纬度的方法（该方法未定义循环，仅能单个坐标转换）
def xy_to_coor(x, y):
    lonlat_coordinate = []
    L = 6381372 * math.pi*2
    W = L
    H = L/2
    mill = 2.3
    lat = ((H/2-y)*2*mill)/(1.25*H)
    lat = ((math.atan(math.exp(lat))-0.25*math.pi)*180)/(0.4*math.pi)
    lon = (x-W/2)*360/W
    # TODO 最终需要确认经纬度保留小数点后几位
    lonlat_coordinate.append(np.around(lon, 8), np.around(lat, 8))
    return lonlat_coordinate

def coordgps2xy(gps_data):
    # gps_data: list of tuple (lon, lat)
    xy_data = []
    for gps in gps_data:
        gps = np.array(gps)
        lon = gps[:, 0]
        lat = gps[:, 1]
        xy = millerToXY(lon, lat)
        xy_data.append(xy)
    return xy_data

def get_external_circle(gps_data):
    # xy_data: list of np.array (n, 2)
    external_circle_info = []
    counter = 0
    for gps in gps_data:
        # 创建一个凸包
        convex_hull = sg.MultiPoint(gps).convex_hull
        # 凸包最小旋转矩形
        bounding_rectangle = convex_hull.minimum_rotated_rectangle
        # 最小旋转矩形的中心点,计算最小半径??
        center = bounding_rectangle.centroid.coords[0]
        radius = max(Point(center).distance(Point(gps[i])) for i in range(len(gps)))
        center = list(center)
        external_circle_info.append([center, radius])    # circle_info: list of [center, radius]
        counter += 1
        if counter % 1000 == 0:
            print("Processing %d / %d" % (counter, len(gps_data)))
    return external_circle_info

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    print(os.getcwd()) # /home/yangshuaiyu6791/RepresentAttack/inversion_dist
    current_path = os.getcwd()
    gps_path = 'data/START/gps/traingps'
    save_path = 'data/START/circle/traincircle'
    gps_data = pickle.load(open(os.path.join(current_path, gps_path), 'rb'))
    
    # # geographic coordinate system to xy coordinate system by miller projection
    # print("Converting...")
    # xy_data = coordgps2xy(gps_data)    # meter as unit

    # just use the geographic coordinate system to get the external circle 
    print("Getting external circle...")
    external_circle_info = get_external_circle(gps_data)

    # Save data
    print("Saving data...")
    pickle.dump(external_circle_info, open(os.path.join(current_path, save_path), 'wb'))
