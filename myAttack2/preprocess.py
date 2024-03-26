import random
import numpy as np
import pickle

# 此文件对轨迹进行预处理
# 删除超过设定的经纬度范围的轨迹 删去去重后长度不符合要求的轨迹
# 保存三个文件，分别为GPS轨迹、去重后的GPS轨迹、去重后的XY坐标轨迹

# 此处设定考虑的区域范围
porto_lat_range = [40.953673,41.307945]
porto_lon_range = [-8.735152,-8.156309]

# 预处理器
class Preprocesser(object):
    def __init__(self, delta = 0.005, lat_range = [1,2], lon_range = [1,2]):
        # 获取delta、纬度范围和经度范围
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self._init_grid_hash_function()

    # 这边实现了对空间的划分，即给定x的取值和y的取值
    def _init_grid_hash_function(self):

        dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
        x  = self._frange(dXMin,dXMax, self.delta) # 返回的是一个列表，里面包含了所有的x取值
        y  = self._frange(dYMin,dYMax, self.delta)
        self.x = x  # interval of x
        self.y = y  # interval of y

    def _frange(self, start, end=None, inc=None):  # divide the range with step = inc =0.005
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
         
        L = []
        while 1:
            next = start + len(L) * inc
            # 超范围则结束循环
            if inc > 0 and next >= end:
                break

            # inc不符合规定则结束循环
            elif inc < 0 and next <= end:
                break

            # 向L中添加新的点
            L.append(next)
        return L

    # 返回值：x坐标，y坐标，和格子编号（从0开始）
    def point2XYandCell(self, tuple):  # return the x,y and index
        test_tuple = tuple # 经纬度最大值
        test_x,test_y = test_tuple[0],test_tuple[1] 
        
        x_grid = int ((test_x-self.lon_range[0])/self.delta) # x坐标
        y_grid = int ((test_y-self.lat_range[0])/self.delta) # y坐标
        index = (y_grid)*(len(self.x)) + x_grid # y*横向格子数量+x坐标 = 格子编号（从0开始）
        return x_grid,y_grid, index  

    # 对GPS轨迹进行去重
    # 返回值：
    #           isCoordinate = False 返回去重之后的cell序列，
    #           isCoordinate = true  返回去重之后的gps序列
    def delDup(self, trajs = [], isCoordinate = False):  # get the index sequence or the coordinate sequence
        # 将每个点替换为格子编号，grid_traj中存放的是轨迹对应的格子编号序列
        grid_traj = []

        # 对于轨迹中的每个点
        for r in trajs:
            # 获取x坐标，y坐标，和格子编号
            _, _, index = self.point2XYandCell((r[2],r[1]))
            grid_traj.append(index)
        
        privious = None  # used to removed the same grid
        hash_traj = []

        # 删去连续的重复的cell/在同一个cell里的点
        for index, i in enumerate(grid_traj):
            if privious==None:
                privious = i
                if isCoordinate == False:
                    hash_traj.append(i)
                elif isCoordinate == True:
                    hash_traj.append(trajs[index][1:])
            else:
                if i==privious:
                    pass
                else:
                    if isCoordinate == False:
                        hash_traj.append(i)
                    elif isCoordinate == True:
                        hash_traj.append(trajs[index][1:])
                    privious = i
        return hash_traj
    
    # 返回值：
    #           isCoordinate = False 返回去重之后的cell序列组成的序列集合，
    #           isCoordinate = true  返回去重之后的gps序列组成的轨迹集合
    def trajsDelDup(self, lenNeedGpsTrajs, isCoordinate =False):  # get the length-satisfied index/coordinate sequence
        trajs_hash = []
        trajs_keys = lenNeedGpsTrajs.keys()  # 获取长度符合要求的轨迹编号
        for key in trajs_keys:
            traj = lenNeedGpsTrajs[key]
            trajs_hash.append(self.delDup(traj, isCoordinate))
        return trajs_hash

    # 返回值：
    #           cell序列/GPS序列、被使用的Cell、去重后的序列最大长度
    def preprocess(self, lenNeedGpsTrajs, isCoordinate = False):
        # 获取去重后的cell序列、被使用的cell和去重后cell序列的最大长度
        if isCoordinate == False:
            cellSeqs = self.trajsDelDup(lenNeedGpsTrajs)
            print("cell序列的总数为：", len(cellSeqs))

            usedCells = {} # 存储了所有被使用的cell
            count = 0  # 去重后序列的总长度
            maxCellSeqLen = 0 # 去重后序列的最大长度

            for cellSeq in cellSeqs:
                if len(cellSeq) > maxCellSeqLen: maxCellSeqLen = len(cellSeq)
                count += len(cellSeq)
                for cell in cellSeq:
                    if usedCells.has_key(cell):
                        usedCells[cell][1] += 1
                    else:
                        usedCells[cell] = [len(usedCells) + 1, 1]
            print("被使用的cell的数量：", len(usedCells.keys()))
            print("去重后序列的总长度：", count, "去重后序列的最大长度：", maxCellSeqLen)
            return cellSeqs, usedCells, maxCellSeqLen
        
        # 获取去重后的gps序列、被使用的cell和去重后cell序列的最大长度
        elif isCoordinate == True:
            delDupGpsSeqs = self.trajsDelDup(lenNeedGpsTrajs, isCoordinate = True)
            maxGpsSeqLen = 0
            usedCells = {}
            for gpsSeq in delDupGpsSeqs:
                if len(gpsSeq) > maxGpsSeqLen: maxGpsSeqLen = len(gpsSeq)
            return delDupGpsSeqs, usedCells, maxGpsSeqLen  # the length-satisfied coordinate sequence, {}, max_len



def trajectory_feature_generation(path ='./rdata/gps_seqs/traingps',
                                  lat_range = porto_lat_range,  # 维度范围
                                  lon_range = porto_lon_range,  # 经度范围
                                  min_length=20):               # 最短轨迹长度
    # fname = path.split('/')[-1].split('_')[0]  # get the city name
    fname = path.split('/')[-1] # 获取文件名

    # 加载轨迹数据
    with open(path, 'rb') as f:
        delDupGpsSeqs = pickle.load(f, encoding = 'iso-8859-1')
    
    lenNeedGpsTrajs = {}  # key为数据集中的index，value为符合要求的[0, 纬度, 经度]GPS轨迹
    maxSeqLen = 0
    preprocessor = Preprocesser(delta = 0.001, lat_range = lat_range, lon_range = lon_range)  # instantiate a Preprocessor
    
    # 输出的是横向的格子数量-1，纵向的格子数量-1，总的格子数量-1
    print("横向格子数量-1，纵向格子数量-1，总的格子数量-1：", preprocessor.point2XYandCell((lon_range[1],lat_range[1])))  # print the lower right corner's x, y and index
    
    for i, traj in enumerate(delDupGpsSeqs):
        new_traj = []  # 用于存储[0, 纬度, 经度]的GPS轨迹
        coor_traj = []  # to store the coordinate sequence
        
        if (len(traj) >= min_length):
            inrange = True
            for p in traj:
                lon, lat = p[0], p[1]
                if not ((lat > lat_range[0]) & (lat < lat_range[1]) & (lon > lon_range[0]) & (lon < lon_range[1])):  # whether in range
                    inrange = False
                    print("第{}范围的".format(i), "超范围的GPS点：", lon, lat)
                new_traj.append([0, p[1], p[0]]) # new_traj存储轨迹，每个轨迹点是[0, 纬度, 经度]的形式

            if inrange:
                # 获取去重之后的GPS轨迹
                coor_traj = preprocessor.delDup(new_traj, isCoordinate=True)
            
                if len(coor_traj)==0:
                    print("第{}条轨迹去重后长度为0".format(i))
                
                # 去重后长度在1~150间的轨迹
                if ((len(coor_traj) >= 1) & (len(coor_traj)<150)):
                    if len(traj) > maxSeqLen: maxSeqLen = len(traj)  # 获取原轨迹最大长度
                    lenNeedGpsTrajs[i] = new_traj 

        if (i+1) %2000==0:
            print("已处理{}条".format(i+1), "其中{}条符合要求".format(len(lenNeedGpsTrajs.keys()))) 
 
    print("符合要求的原轨迹最大长度为：", maxSeqLen)
    print("最终一共有{}条轨迹符合要求(不超范围，去重后长度符合要求)".format(len(lenNeedGpsTrajs.keys())))

    pickle.dump(lenNeedGpsTrajs, open('./features/{}_lenNeedGpsTrajs'.format(fname),'wb'))  # serialize the length-needed trajs(coordinate) {i: [0,lat,lon]...}
    print("轨迹筛选完毕，范围和长度符合条件的轨迹存储于\'./features/{}_lenNeedGpsTrajs\'".format(fname))

    delDupGpsSeqs, usedCells, maxSeqLen = preprocessor.preprocess(lenNeedGpsTrajs, isCoordinate=True)  # the length-satisfied coordinate sequences, {}, max_len
    
    pickle.dump((delDupGpsSeqs,[],maxSeqLen), open('./features/{}_delDupGpsSeqs'.format(fname), 'wb'))  # serialize the length-needed trajs(coordinate) {[lat,lon]...}
    print("轨迹去重完毕，去重后的GPS轨迹存储于\'./features/{}_delDupGpsSeqs\'".format(fname))

    # traj_grids = cPickle.load(open('./data_taxi/porto_traj_coord'))
    # 找出去重后符合要求的轨迹的最小/最大x, y坐标，来框定所有轨迹实际经过的范围
    xySeqs = []
    # min_x, min_y, max_x, max_y = 2000, 2000, 0, 0
    # for i in delDupGpsSeqs:
    #     for j in i:
    #         x, y, cell = preprocessor.point2XYandCell((j[1], j[0]))
    #         if x < min_x:
    #             min_x = x
    #         if x > max_x:
    #             max_x = x
    #         if y < min_y:
    #             min_y = y
    #         if y > max_y:
    #             max_y = y
    # print("最小x，y和最大x，y：", min_x, min_y, max_x, max_y)  # update the max/min covered index

    # 将轨迹转换为x, y坐标序列，此处的x,y坐标是采用最大、最小坐标框定后的新的范围。
    for i in delDupGpsSeqs:
        xySeq = []
        for j in i:
            x, y, cell = preprocessor.point2XYandCell((j[1], j[0]))
            x = x
            y = y
            xy = [y, x]
            xySeq.append(xy)
        xySeqs.append(xySeq)  # update the new index sequence relative to the cover index
    

    print(len(xySeqs))
    
    pickle.dump((xySeqs,[],maxSeqLen), open('./features/{}_xySeqs'.format(fname), 'wb'))  # serialize the length-needed trajs(grid index) {[y,x]...}
    print("轨迹对应的xy坐标序列存储于：\'./features/{}_xySeqs\'".format(fname))
    
    # 返回值：去重后的gps轨迹、文件名
    return './features/{}_delDupGpsSeqs'.format(fname), fname