bev pool 流程

1.将camera feature和depth feature进行内积，得到camera feature point cloud。

2.将camera feature point cloud里的所有点投影到bev grid里(对于越界的不投影)

3.将一个bev grid里面的所有点进行相加

4.最终得到[1,c,bh,bw]维度的bev feature，用于后续处理

BEVfusion和CUDA-BEVfusion中对这一部分进行了加速

1.precomputation

2.interval reduction