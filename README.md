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


### The overall program is form of the following 10 components (with specific algorithm):

* Convolution.py
  * Padding Algorithm 
  * Convolution Algorithm


* Convolution_anchor.py
  * Padding algorithm with Specific Anchor
  * Convolution Algorithm with Specific Anchor


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
  
 
### 整体代码结构由以下10个部分所构成（附带具体算法）：

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
双边掩膜(滤波): [OpenCV 学习：９ 双边滤波bilateralFilter](https://zhuanlan.zhihu.com/p/127023952)\
经典算子(滤波):  [【图像处理】轻松搞懂图像锐化](https://zhuanlan.zhihu.com/p/162275458)\
按位运算: [OpenCV 之按位运算举例解析](https://blog.csdn.net/qq_36758914/article/details/106836231)\
腐蚀、膨胀、开运算、闭运算: [图像处理：图像腐蚀、膨胀，开操作、闭操作](https://codeantenna.com/a/mYT2rbm2Q9)\
形态学梯度变化、顶帽变化: [形态学处理（腐蚀膨胀，开闭运算，礼帽黑帽，边缘检测](https://www.cnblogs.com/wj-1314/p/12084636.html)\
RGB与HSV 格式转化: [色彩转换系列之RGB格式与HSV格式互转原理及实现](https://blog.csdn.net/weixin_40647819/article/details/92660320)\
3D颜色查找表转化: [LUT（look up table）调色的原理与代码实现](https://www.jianshu.com/p/d09aeea3b732)\
大津法阈值分割算法: [otsu阈值分割算法原理_Opencv从零开始](https://blog.csdn.net/weixin_35943182/article/details/112443343)\
三角法阈值分割算法: [图像处理之三角法图像二值化](https://blog.csdn.net/jia20003/article/details/53954092)\
自适应阈值处理: [灰度图像-图像分割](https://face2ai.com/DIP-7-7-%E7%81%B0%E5%BA%A6%E5%9B%BE%E5%83%8F-%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2-%E9%98%88%E5%80%BC%E5%A4%84%E7%90%86%E4%B9%8B%E5%B1%80%E9%83%A8%E9%98%88%E5%80%BC)\
Canny 算法: [Canny边缘检测](https://www.cnblogs.com/mmmmc/p/10524640.html)\
轮廓提取算法: [OpenCV轮廓提取算法详解findContours()](https://zhuanlan.zhihu.com/p/107257870)\

### Relevant Paper:
* [Triangle Optimal Threshold: Automatic measurement of sister chromatid exchange](https://pubmed.ncbi.nlm.nih.gov/70454/)
* [BGR and Gray-scale image conversion: Decolorize-fast, contrast enhancing color to grayscale conversion](https://journals.sagepub.com/doi/pdf/10.1177/25.7.70454)
* [Skin Color detection: RGB-H-CbCr skin colour model for human face detection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf)
* [Find Contour Algorithm: Topological Structural analysis of digitized binary images by border following](https://www.sciencedirect.com/science/article/abs/pii/0734189X85900167)
  
### 相关文献:
* 三角性最优阈值查找: [Automatic measurement of sister chromatid exchange](https://pubmed.ncbi.nlm.nih.gov/70454/)
* BGR颜色格式和灰色图片格式转化: [Decolorize-fast, contrast enhancing color to grayscale conversion](https://journals.sagepub.com/doi/pdf/10.1177/25.7.70454)
* 皮肤颜色追踪: [RGB-H-CbCr skin colour model for human face detection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf)
* 轮廓查找: [Topological Structural analysis of digitized binary images by border following](https://www.sciencedirect.com/science/article/abs/pii/0734189X85900167)


## Result and algorithm comparison(结果以及算法比较)

All of the transformed image is saved in result file.(所有转化后的图像都保存在result 文件夹中)

* Algorithm comparison:
 * Interpolation Algorithm: Bicubic Algorithm > Bilinear Algorithm > Nearest-neightbor Algorithm
 * Filtering Algorithm: Bilateral Filtering Algorithm > Gaussian Filtering Algorithm
 * Noising Removing Threshold Algorithm: Gaussian-filtering Otsu threshold > Gaussian-Filtering adaptive threshold > Bilateral-filtering Otsu threshold
 * Edge Detection: Canny Algorithm > Sobel Algorithm > Laplace Algorithm

* 算法对比：
 * 插值算法: 三次样条插值算法 > 双线性插值算法 > 最邻近插值算法
 * 掩膜(滤波)算法：双边滤波算法 > 高斯滤波算法
 * 带有噪声处理的阈值处理: 高斯过滤-大津阈值算法 > 高斯过滤-自适应阈值处理 > 双边滤波大津阈值处理
 * 边缘检测: Canny 算法 > 索贝尔算法 > 拉普拉斯算法



## Author(关于作者)

Jianfan Shao – Jinan University\
E-mail: jackshaw0714@gmail.com\
Note: Please send an email for permission to use the appeal code for educational or commercial purposes



邵键帆-暨南大学\
邮箱: jackshaw0714@gmail.com\
注意：如需将该项目作为教学或者商业用途，请发邮件征得本人同意


