# -*- coding: utf-8 -*-

from yolo.predict import YOLO_V2_Predict
from yolo.labels import mylabels
import sys,getopt

image = "image/t.jpg"
pbfile = 'pb/yolo.pb'
metafile = 'pb/yolo.meta'
thresh = 0.2
savedir = 'image/predict.jpg'
labels = None

def useage():
    print('test.py -i <image file> -p <xxx.pb file> -m <xxx.meta file> -t <thresh> -d <save_dir> -l')
    print('  -h 或 --help:     帮助信息')
    print('  -i 或 --image:    待预测的图片名称及路径')
    print('  -p 或 --pbfile:   .pb模型文件路径及名称')
    print('  -m 或 --metafile: .meta模型文件路径及名称')
    print('  -t 或 --thresh:   设置阈值大小 0到1之间的数')
    print('  -d 或 --savedir:  预测完成后的图片保存的名称及路径')
    print('  -l:               是否按照设定限制输出类型')

try:
    opts, args = getopt.getopt(sys.argv[1:],"h:i:p:m:t:d:l",
            ["help=","image=","pbfile=","metafile=","thresh=","savedir="])
except getopt.GetoptError:
    useage()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-h", "--help"):
        useage()
        sys.exit()
    elif opt in ("-i", "--image"):
        image = arg
    elif opt in ("-p", "--pbfile"):
        pbfile = arg
    elif opt in ("-m", "--metafile"):
        metafile = arg
    elif opt in ("-t", "--thresh"):
        thresh = float(arg)
    elif opt in ("-d", "--savedir"):
        savedir = arg
    elif opt == "-l":
        labels = mylabels

#tiny-yolo-voc   yolo
YOLO_V2_Predict(pbfile, metafile ,thresh=thresh,classes=labels).predict(image,save_dir=savedir)


