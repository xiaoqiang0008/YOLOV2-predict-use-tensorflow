# YOLOV2-predict-use-tensorflow
加载tensorflow的pb模型，实现目标检测功能
参考：https://blog.csdn.net/dabudaodexiaoqiang/article/details/80738837

环境要求：

系统：windows7_X64

Tensorflow 1.4

Opencv3

python3.5

先看一下目录结构：



从上往下：

build文件夹存放Cython代码编译的中间文件；

image文件夹存放测试的图片文件；

lib文件夹存放Cython代码相关的模块，该部分主要摘自darkflow项目；

pb文件夹存放训练好的模型文件，包括后缀名为 ’.pb’以及 ’.meta’ 两个文件，前者记录网络结构、权重等变量，后者记录网络的一下参数信息，例如具体类别等；

yolo文件夹存放加载模型文件并实现目标检测的相关模块。

 至于pb模型怎么来的，查看darkflow项目的说明，里面有把darknet模型转换为tensorflow的pb模型的指令，搞一搞就可以了。

使用方法说明：

开始之前需要先编译Cython代码。打开命令行窗口，进入到根目录下，执行

python setup.py build_ext --inplace

如果执行上面命令报错的话，可能是因为你的电脑没有安装Cython以及与Python版本相对应的VS的C编译器，这个搞起来也真是麻烦，不过网上有很多解决办法的。

执行 -h 或 --help 查看使用方法

python main.py -h 或 python main.py --help

只有-i 或 --image 必须要指定之外，其他都是由默认值的，不更改的话，直接采用默认即可
接下来可以试着用命令行检测一张照片，看看效果！

python main.py -i image/car.jpg

预测完成，会默认在image文件夹下生成一个名为predict.jpg的图片文件。

当然，pb文件夹中的yolo模型可以预测输出80目标，然而有时候，只想检测几类目标，这样可以利用-l指令了。加上-l的话，程序会自动加载yolo/labels.py 文件下的mylabels列表。我们可以修改这个列表的值，实现只输出制定目标类型。下面看看效果！

先后执行

python main.py -i image/car.jpg
python main.py -i image/car.jpg -l

 我的yolo/labels.py中mylabels只设置了一个类别“person”,所以第一张图片只显示检测到了两个人，其他车辆等目标没有显示。

使用命令行模式实现预测是为了测试方便，如果想直接嵌入到自己的项目里，可以修改main.py文件，像这样子：

from yolo.predict import YOLO_V2_Predict
from yolo.labels import mylabels
import cv2

image = "image/t.jpg"
pbfile = 'pb/yolo.pb'
metafile = 'pb/yolo.meta'
thresh = 0.2
savedir = 'image/predict.jpg'
labels = mylabels

yolo = YOLO_V2_Predict(pbfile, metafile ,thresh=thresh,classes=labels)
img = yolo.predict(image,save=False)[0]
cv2.imshow('predict',img)
cv2.waitKey(0)
