## Opencv Library basic graphic processing algorithm coding reproduction based on Numpy and Matplotlib library (基于Numpy和Matplotlib 库对于Opencv基础图像处理算法代码复现)

This repository simulated the basic graphic image processing function through reproduce the underlying algorithm based on numpy and matplotlib library. This project is aiming at in-depth understanding of image-processing algorithm and serves for the introduction to Computer Vision (CV). (这个仓库在不依赖Opencv 库的前提下，利用python基础Numpy 和 matplotlib 库编程实现Opencv 图像处理的基础功能，这个项目旨在于对于深入了解图像处理的底层算法，和作为计算机视觉的入门知识储备)

Currently, this repository contained Convolution Transformation\ Image Interpolation(resize) Transformation \ Classical Filter Transformation \ Image Arithmetic Operation \ Color Transformation \ Threshold Segmentation \ Edge Detection \ Contour detection basic image transformation section. (目前，这个仓库涵盖了基础的卷积处理、图像插值处理、经典算子变化、图像算术运算、颜色转化、阈值处理、边缘检测、轮廓检测的基础图像转化板块)


![Example of Lenna](result/lenna_display.png)

## Library Dependency(依赖包)

1.	Python 
2.	Numpy
3.	Matplotlib
4.	Pandas
5.	Math

## Installation(安装包)

It is better to create new env for this new project to avoid the Incompatibility(最好创建一个新的python环境运行cv项目，以避免与其他项目包出现兼容问题)

```bash
pip install numpy 
pip install matplotlib
pip install pandas
```

## Run Test Section(测试运行)

At the end of each .py file, there is the testing section. When you remove the triple comment quotes, you can run the code directly. (测试模块位于每一个python文件的末端，在测试模块中直接移除上下三引号注释符可以直接运行代码)

```bash

##################################### TEST ####################################
“””
Main Testing code
“””
##################################### TEST ####################################

```

## Coding and corresponding algorithm structure(代码与对应算法架构)

Most of the coding sections are divided into the one-channel image (gray-scale image) transformation region and three-channel image (BGR-image or RGB-image) transformation region. In this rep, we simulate the Opencv image reading method (BGR-format image) The testing image is saved in pic file.  大部分的代码文件都被分割为一管道图像（大致理解为灰度图像）处理区域和和 三管道图像（BGR-图像或者RBG-图像）处理区域。所有测试的图像均放在pic 文件夹中。

* Convolution.py
  * Padding Algorithm 
  * Convolution Algorithm


* Convolution_anchor.py
  *Padding algorithm with Specific Anchor
  *Convolution Algorithm with Specific Anchor


* Image_resize.py
  * Interpolation algorithm: Nearest-neighbor Interpolation\ Bilinear Interpolation\ Bicubic Interpolation


* Typical_filter.py
  * Overall Filtering: Average Filtering\ Median Filtering\ Maximum Filtering\ Minimum Filtering
  * Local Filtering: Gaussian Filtering\ Bilateral Filtering


* Arithmetic_operation.py
  * Arithmetic Operation: Add Operation\ Subtract Operation
  * Bitwise Operation: Bitwise And Operation\ Bitwise Or Operation\ Bitwise Xor Operation\ Bitwise Not Operation


* Morphological_trans.py
  * Topology Algorithm: Erode Algorithm\ Dilate Algorithm\ Opening Algorithm\ Closing Algorithm\ Gradient Algorithm\ White Top Hat Algorithm\ Black Top Hat Algorithm


* Color_trans.py
  * Image format conversion algorithm: BGR and Gray-scale Image Conversion Algorithm\ BGR and HSV Conversion Algorithm 
  * 3D Look-Up-Table (LUT) color transformation


* Threshold_segmentation.py
  * Overall Threshold Algorithm: Binary Threshold\ Binary Inverse Threshold\ Truncation Threshold\ To-Zero Threshold\ To-Zero Inverse Threshold
  * Local threshold Algorithm: Adaptive Mean threshold \ Adaptive Median threshold \ Adaptive Gaussian Threshold  
  * Optimal Threshold Algorithm: Otsu Optimal Threshold Algorithm\ Triangle Optimal Threshold Algorithm (It’s normal to integrate the Optimal Threshold Algorithm with the Overall Threshold Algorithm)
  * Noise Removing Threshold Algorithm: Gaussian-filtering Adaptive Gaussian Threshold Algorithm\ Gaussian Filtering Otsu Threshold Algorithm\ Bilateral-Filtering Otsu Threshold Algorithm 


* Edge_detection.py
  * Filtering Algorithm: Laplace Transformation Algorithm\ Sobel Gradient Transformation Algorithm 
  * Nosie Filtering + Threshold suppression Algorithm: Canny Algorithm


* Contour_detection.py
  * Topology Algorithm: Topological Structural Analysis 
  
 
整体代码结构由以下10个部分所构成（附带具体算法）：

* Convolution.py
  * 填充算法 
  * 卷积算法


* Convolution_anchor.py
  * 带特定锚点的填充算法
  * 带特定锚点的卷积算法


* Image_resize.py
  * 内插算法: 最邻近值插值法\ 双线性插值法\ 三次样条插值法


* Typical_filter.py
  * 全局掩膜算法： 平均掩膜算法\ 中位值掩膜算法\ 最大值掩膜算法\ 最小值掩膜算法
  * 局部掩膜算法 (可变掩膜算法)：高斯掩膜算法\ 双边掩膜算法


* Arithmetic_operation.py 
  * 图像算术运算: 图像算术加减运算
  * 图像按位运算: 图像按位和运算\ 图像按位或运算\ 图像按位异或运算\ 图像非运算


* Morphological_trans.py
  * 拓扑算法: 腐蚀算法\ 膨胀算法\ 开运算\ 闭运算\ 梯度算法\ 白顶帽算法\ 黑顶帽算法


* Color_trans.py
  * 图像颜色格式转化算法: BGR格式和灰度格式相互转化\ BGR格式和HSV 格式相互转化算法
  * 3D 颜色查找表颜色转化算法


* Threshold_segmentation.py
  * 全局阈值算法: 二值阈值算法\ 反二值阈值算法\ 截断阈值算法\ 零化阈值算法\ 反零化阈值算法
  * 局部阈值算法 (自适应阈值算法): 自适应均值均值算法\ 自适应中位数阈值算法\ 自适应高斯阈值算法
  * 优化阈值查找算法: 大津法优化阈值算法\ 三角形优化阈值算法 (通过找到局部最优的阈值与全局阈值算法结合，从而对于图像进行阈值处理)
  * 带有噪声过滤的阈值处理算法: 高斯过滤-自适应阈值算法\ 高斯过滤-大津优化阈值算法\ 双边过滤-大津优化阈值处理


* Edge_detection.py
  * 基于算子的边缘检测: 拉普拉斯转化算法\ 索贝尔梯度转化算法
  * 基于噪声过滤算子和阈值抑制算法: Canny 算法


* Contour_detection.py
  * 拓扑算法： 领域拓扑结构分析算法


## Reference and Learning Material(参考文献以及学习资料)

Learning material of fundamental algorithm and models such as image convolution is easy accessible in the open Internet. In this section, I only provide more sophisticated algorithm material. (基础模型和算法，类似图形卷积处理这种在公开网络很容易获取对应的资料，因此在这个部分我们将提供略微有点难度的算法内容)


### Blog material (Only in Chinese) 博客资料(仅提供中文)
双线性插值: [图像处理+双线性插值法](https://blog.csdn.net/lovexlsforever/article/details/79508602)\
三次样条插值:[最近邻插值、双线性插值、双三次插值](https://blog.csdn.net/caomin1hao/article/details/81092134)\
高斯掩膜(滤波): [OpenCV 学习：8 高斯滤波GaussianBlur](https://zhuanlan.zhihu.com/p/126592928)\



  

