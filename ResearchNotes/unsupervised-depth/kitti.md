# KITTI

分析 KITTI 数据集的结构以及数据传输方式。

## 数据集结构

```
/mnt/disk/kitti/
├── 2011_09_26
│   ├── 2011_09_26_drive_0001_sync
│   ├── 2011_09_26_drive_0002_sync
│   ├── 2011_09_26_drive_0005_sync
│   ├── 2011_09_26_drive_0009_sync
│   ├── 2011_09_26_drive_0011_sync
│   ├── 2011_09_26_drive_0013_sync
│   ├── 2011_09_26_drive_0014_sync
│   ├── 2011_09_26_drive_0015_sync
│   ├── 2011_09_26_drive_0017_sync
│   ├── 2011_09_26_drive_0018_sync
│   ├── 2011_09_26_drive_0019_sync
│   ├── 2011_09_26_drive_0020_sync
│   ├── 2011_09_26_drive_0022_sync
│   ├── 2011_09_26_drive_0023_sync
│   ├── 2011_09_26_drive_0027_sync
│   ├── 2011_09_26_drive_0028_sync
│   ├── 2011_09_26_drive_0029_sync
│   ├── 2011_09_26_drive_0032_sync
│   ├── 2011_09_26_drive_0035_sync
│   ├── 2011_09_26_drive_0036_sync
│   ├── 2011_09_26_drive_0039_sync
│   ├── 2011_09_26_drive_0046_sync
│   ├── 2011_09_26_drive_0048_sync
│   ├── 2011_09_26_drive_0051_sync
│   ├── 2011_09_26_drive_0052_sync
│   ├── 2011_09_26_drive_0056_sync
│   ├── 2011_09_26_drive_0057_sync
│   ├── 2011_09_26_drive_0059_sync
│   ├── 2011_09_26_drive_0060_sync
│   ├── 2011_09_26_drive_0061_sync
│   ├── 2011_09_26_drive_0064_sync
│   ├── 2011_09_26_drive_0070_sync
│   ├── 2011_09_26_drive_0079_sync
│   ├── 2011_09_26_drive_0084_sync
│   ├── 2011_09_26_drive_0086_sync
│   ├── 2011_09_26_drive_0087_sync
│   ├── 2011_09_26_drive_0091_sync
│   ├── 2011_09_26_drive_0093_sync
│   ├── 2011_09_26_drive_0095_sync
│   ├── 2011_09_26_drive_0096_sync
│   ├── 2011_09_26_drive_0101_sync
│   ├── 2011_09_26_drive_0104_sync
│   ├── 2011_09_26_drive_0106_sync
│   ├── 2011_09_26_drive_0113_sync
│   ├── 2011_09_26_drive_0117_sync
│   ├── 2011_09_26_drive_0119_sync
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
├── 2011_09_28
│   ├── 2011_09_28_drive_0001_sync
│   ├── 2011_09_28_drive_0002_sync
│   ├── 2011_09_28_drive_0016_sync
│   ├── 2011_09_28_drive_0021_sync
│   ├── 2011_09_28_drive_0034_sync
│   ├── 2011_09_28_drive_0035_sync
│   ├── 2011_09_28_drive_0037_sync
│   ├── 2011_09_28_drive_0038_sync
│   ├── 2011_09_28_drive_0039_sync
│   ├── 2011_09_28_drive_0043_sync
│   ├── 2011_09_28_drive_0045_sync
│   ├── 2011_09_28_drive_0047_sync
│   ├── 2011_09_28_drive_0053_sync
│   ├── 2011_09_28_drive_0054_sync
│   ├── 2011_09_28_drive_0057_sync
│   ├── 2011_09_28_drive_0065_sync
│   ├── 2011_09_28_drive_0066_sync
│   ├── 2011_09_28_drive_0068_sync
│   ├── 2011_09_28_drive_0070_sync
│   ├── 2011_09_28_drive_0071_sync
│   ├── 2011_09_28_drive_0075_sync
│   ├── 2011_09_28_drive_0077_sync
│   ├── 2011_09_28_drive_0078_sync
│   ├── 2011_09_28_drive_0080_sync
│   ├── 2011_09_28_drive_0082_sync
│   ├── 2011_09_28_drive_0086_sync
│   ├── 2011_09_28_drive_0087_sync
│   ├── 2011_09_28_drive_0089_sync
│   ├── 2011_09_28_drive_0090_sync
│   ├── 2011_09_28_drive_0094_sync
│   ├── 2011_09_28_drive_0095_sync
│   ├── 2011_09_28_drive_0096_sync
│   ├── 2011_09_28_drive_0098_sync
│   ├── 2011_09_28_drive_0100_sync
│   ├── 2011_09_28_drive_0102_sync
│   ├── 2011_09_28_drive_0103_sync
│   ├── 2011_09_28_drive_0104_sync
│   ├── 2011_09_28_drive_0106_sync
│   ├── 2011_09_28_drive_0108_sync
│   ├── 2011_09_28_drive_0110_sync
│   ├── 2011_09_28_drive_0113_sync
│   ├── 2011_09_28_drive_0117_sync
│   ├── 2011_09_28_drive_0119_sync
│   ├── 2011_09_28_drive_0121_sync
│   ├── 2011_09_28_drive_0122_sync
│   ├── 2011_09_28_drive_0125_sync
│   ├── 2011_09_28_drive_0126_sync
│   ├── 2011_09_28_drive_0128_sync
│   ├── 2011_09_28_drive_0132_sync
│   ├── 2011_09_28_drive_0134_sync
│   ├── 2011_09_28_drive_0135_sync
│   ├── 2011_09_28_drive_0136_sync
│   ├── 2011_09_28_drive_0138_sync
│   ├── 2011_09_28_drive_0141_sync
│   ├── 2011_09_28_drive_0143_sync
│   ├── 2011_09_28_drive_0145_sync
│   ├── 2011_09_28_drive_0146_sync
│   ├── 2011_09_28_drive_0149_sync
│   ├── 2011_09_28_drive_0153_sync
│   ├── 2011_09_28_drive_0154_sync
│   ├── 2011_09_28_drive_0155_sync
│   ├── 2011_09_28_drive_0156_sync
│   ├── 2011_09_28_drive_0160_sync
│   ├── 2011_09_28_drive_0161_sync
│   ├── 2011_09_28_drive_0162_sync
│   ├── 2011_09_28_drive_0165_sync
│   ├── 2011_09_28_drive_0166_sync
│   ├── 2011_09_28_drive_0167_sync
│   ├── 2011_09_28_drive_0168_sync
│   ├── 2011_09_28_drive_0171_sync
│   ├── 2011_09_28_drive_0174_sync
│   ├── 2011_09_28_drive_0177_sync
│   ├── 2011_09_28_drive_0179_sync
│   ├── 2011_09_28_drive_0183_sync
│   ├── 2011_09_28_drive_0184_sync
│   ├── 2011_09_28_drive_0185_sync
│   ├── 2011_09_28_drive_0186_sync
│   ├── 2011_09_28_drive_0187_sync
│   ├── 2011_09_28_drive_0191_sync
│   ├── 2011_09_28_drive_0192_sync
│   ├── 2011_09_28_drive_0195_sync
│   ├── 2011_09_28_drive_0198_sync
│   ├── 2011_09_28_drive_0199_sync
│   ├── 2011_09_28_drive_0201_sync
│   ├── 2011_09_28_drive_0204_sync
│   ├── 2011_09_28_drive_0205_sync
│   ├── 2011_09_28_drive_0208_sync
│   ├── 2011_09_28_drive_0209_sync
│   ├── 2011_09_28_drive_0214_sync
│   ├── 2011_09_28_drive_0216_sync
│   ├── 2011_09_28_drive_0220_sync
│   ├── 2011_09_28_drive_0222_sync
│   ├── 2011_09_28_drive_0225_sync
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
├── 2011_09_29
│   ├── 2011_09_29_drive_0004_sync
│   ├── 2011_09_29_drive_0026_sync
│   ├── 2011_09_29_drive_0071_sync
│   ├── 2011_09_29_drive_0108_sync
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
├── 2011_09_30
│   ├── 2011_09_30_drive_0016_sync
│   ├── 2011_09_30_drive_0018_sync
│   ├── 2011_09_30_drive_0020_sync
│   ├── 2011_09_30_drive_0027_sync
│   ├── 2011_09_30_drive_0028_sync
│   ├── 2011_09_30_drive_0033_sync
│   ├── 2011_09_30_drive_0034_sync
│   ├── 2011_09_30_drive_0072_sync
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
├── 2011_10_03
│   ├── 2011_10_03_drive_0027_sync
│   ├── 2011_10_03_drive_0034_sync
│   ├── 2011_10_03_drive_0042_sync
│   ├── 2011_10_03_drive_0047_sync
│   ├── 2011_10_03_drive_0058_sync
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
└── readme.txt
```

数据集按照：一、日期；二、drive 的方式组织起来。

每个 drive 结构如下：

```
/mnt/disk/kitti/2011_09_26/2011_09_26_drive_0001_sync/
├── image_00 # 灰度图
│   ├── data # *.png
│   └── timestamps.txt
├── image_01 # 灰度图
│   ├── data # *.png
│   └── timestamps.txt
├── image_02 # 左相机图
│   ├── data # *.png
│   └── timestamps.txt
├── image_03 # 右相机图
│   ├── data # *.png
│   └── timestamps.txt
├── oxts
│   ├── data # *.txt
│   ├── dataformat.txt
│   └── timestamps.txt
└── velodyne_points # 雷达点云数据
    ├── data # *.bin
    ├── timestamps_end.txt
    ├── timestamps_start.txt
    └── timestamps.txt
```

## 数据传输方式

这里借用 monodepth2 的 dataset 文件作为参考：



