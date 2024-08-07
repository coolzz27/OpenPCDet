# 训练流程

## rosbag 录制

1. IPv4 设置 192.168.1.50，网线连接 livox mid70
2. ```cd ~/ws_livox/ && source devel/setup.sh```
3. ```roslaunch livox_ros_driver livox_lidar_rviz.launch```
4. ```rosbag record -O outpost.bag /livox/lidar```

## 数据集准备

1. bag to pcd:

    ```rosrun pcl_ros bag_to_pcd outpost.bag /livox/lidar outpost_pcd```

2. 标注:

    ```roslaunch annotate demo.launch bag:="/home/vioplum-hp/rosbag/outpost.bag --pause-topics /livox/lidar"```

    - 修改```Fixed Frame```为```livox_frame```
    - 修改```Topic```为```/livox/lidar```

3. 标注结果转txt：

    从```~/.ros```中找到```annotate.yaml```，使用```python3 yaml_to_txt.py```

4. 筛选标注好的pcd文件：

    使用```python3 copy_pcd.py```将```outpost_pcd```路径中被标注的```pcd```文件复制到```labeled_pcd```文件夹中

5. pcd to npy:

    使用```python3 pcd_to_npy.py```将```pcd```文件转换为```npy```文件

## 使用 OpenPCDet 训练

1. 初始化：

     ```python3 setup.py develop``` (如果报错 ```c++ not compatible with pytorch ...```, 则在前面添加 ```CXX=g++```)

2. 训练：

    ```cd tools && python3 train.py --cfg_file cfgs/custom_models/pv_rcnn.yaml```

3. 可视化：

    ```python3 demo.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --data_path ../data/custom/points/ --ext .npy --ckpt ../output/custom_models/pv_rcnn/default/ckpt/checkpoint_epoch_{替换}.pth```

- 可视化一帧pcd点云：```pcl_viewer <file_name>```

## 使用 ros 实时检测

1.
    ```bash
    roscore
    ```

    (terminal1)

2.
    ```bash
    cd ~/ws_livox/ && source devel/setup.sh
    roslaunch livox_ros_driver livox_lidar_rviz.launch
    ```

    (terminal2 用于实时显示点云)

3.
    ```bash
    conda activate openmmlab
    cd ./ros
    python3 pointcloud_saver.py
    ```

    (terminal3 用于保存点云)

4.
    ```bash
    conda activate openmmlab
    cd ./ros
    python3 ros_demo.py
    ```

    (terminal4 用于检测)
