# 店铺名称检测

## 1. 环境依赖

- windows or linux

- python3.7

- pytorch-cpu or pytorch-gpu

- 注：建议手动安装pytorch（pip安装速度慢）

  

## 2. 快速开始

- 首先安装依赖项，根目录下运行：

  ```python
  pip install -r requirments.txt
  ```

  注：shapely库的pip安装易导致**[winRrror 126] 找不到指定模块的问题**，可手动下载whl文件安装

- 依赖安装完毕后，直接运行detector.py，默认变量为待测图片名称:

  ```python
  python3 detector.py test.jpg
  ```

  注：默认变量也可以是图片路径参数或文件夹

- 输出结果为店铺实体名称、置信度与预测耗费时间

## 3. 模型搭建

​		本项目主要分为两个模块，一是基于百度Paddleocr的文字识别模块，二是基于pytorch框架的店铺名实体检测模块。（Paddleocr参见https://github.com/PaddlePaddle/PaddleOCR）

​		OCR文字检测和识别模块使用PaddleOCR集成的DB模型（推荐）和CRNN模型（推荐），且是PaddleOCR当中的轻量级模块（识别快速且效果不错）。识别出的文字内店铺名的检测使用我们自己搭建的模型 - **实体检测模块**。

### 	实体检测模块

​		实体检测模块由图像特征和语义特征联合进行实体的识别。根据两个特征来求解置信度，最后根据置信度进行判断。

​		考虑到目标区域通常占面积较大，且位于图片的中间部分，基于此假设可以通过OCR识别结果的面积得到图像特征，并根据OCR识别结果与图像中心的距离调整面积；语义特征的提取选用二分类模型来区分店铺名实体和噪声，输入当前图片所有识别文字拼接的结果和当前需要预测的识别结果，输出识别结果的分类和概率。

​		例如对一张图片，OCR识别出了[开心发廊，洗剪吹，烫染，电话]，先拼接起来得到局部语境“开心发廊洗剪吹烫染电话”，然后把局部语境和预测出的结果挨个输入进模型进行判断。

​		其中依靠图像特征的判断更可靠，所以最终置信度加权求和时，图像的prob权重比较大。

## 4. 实验结果

​		以下图所示图片为例：

```python
python detector.py ./pics/test.jpg
```

<img src="https://github.com/TheoRenLi/Corporate-named-entity-recognition/blob/main/pics/test.jpg" style="zoom: 33%;" align="center"/>

​		PaddleOCR识别结果：

<img src="https://github.com/TheoRenLi/Corporate-named-entity-recognition/blob/main/pics/ocr_result.jpg" style="zoom:50%;" align="center"/>

​		店铺名称识别结果：

<img src = "https://github.com/TheoRenLi/Corporate-named-entity-recognition/blob/main/pics/test_result.png" align="center">

## Reference

- https://github.com/PaddlePaddle/PaddleOCR
- 数据来源：第九届“中国软件杯”大学生软件设计大赛；[图片数据：ICDAR2017](http://rctw.vlrlab.net/dataset/)

