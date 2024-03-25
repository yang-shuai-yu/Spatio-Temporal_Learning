### My first try at generating trajectories
A simple attempt by using LDPTrace.

### Steps
1. Get the grids of the map
    这里首先我可以先采用固定大小的grid，然后再根据实际情况进行调整。
2. The LDPTrace server
    Markov是包括在这一部分中的，应该作为类内函数存在
3. The LDPTrace client
    生成轨迹(Trajectory synthesis),同样因该新开一个类，调用LDPTrace server类中的函数
4. Check the result
    评价指标，error以及可视化