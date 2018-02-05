http://m.blog.csdn.net/qq_34484472/article/details/73135354
http://blog.csdn.net/ch_liu23/article/details/53558549
1. yolo 
yolo的官网：https://pjreddie.com/darknet/yolo/ 
yolo的官网介绍了yolo的安装与测试。建议大家多看看英文官网，因为中文网更新的慢，而且有部分内容省略了。按照官网的步骤就不会有错。

2. 数据的预处理 
yolo的数据包括训练数据和验证数据（训练数据用来训练模型，验证数据用来调整模型）。训练数据和验证数据都包括：a.图片；b.标签。需要说明的是，如果采用VCC的话，标签需要特定xml格式，还要转化为txt。下面以我目标检测“猫”为例讲解。
a.在“Image”文件夹下存放所有的图片样本（包括训练数据和验证数据，而且最好是jpg格式） 
b.下载labelImg（一种图像标记工具，给图像中的目标打上标签，且可以生成训练需要的xml格式），具体的使用方法可以百度，操作起来很简单。 
c.与“Image”文件夹同级新建“xml”文件夹，“xml”文件夹存放labelImg得到的所有图片样本的标签。 
d.现在就是要将所有的样本分成训练集和验证集，而且要将训练集和验证集对应的xml也分开。这里下载python脚本，直接放在“Image”和“xml”文件夹同级路径。

如果上面无法下载，试试： 
链接：http://pan.baidu.com/s/1hs22I7U 密码：wdv0 
运行traindata.py：生成trainImage文件夹，存放训练图片；生成trainImageXML文件夹，存放训练图片xml标签；生成validateImage文件夹，存放验证集图片；生成validateImageXML文件夹，存放验证集图片的xml标签。 
运行trans.py，生成trainImageLabelTxt文件夹，存放训练图片通过xml标签转化得到的txt文件（若在训练过程提示txt文件找不到，则把此文件夹下的txt文件夹移动到trainImage文件夹）；生成validateImageLabelTxt文件夹，道理一样。 
另外得到的trainImagePath.txt和validateImagePath.txt存放着训练图片和验证图片的路径。

3. 修改配置文件 
接下来就是修改配置文件了： 
cfg/voc.data文件中: 
classes= 1 
train = /home/pdd/pdwork/darknet2/darknet/VOC/cat/trainImagePath.txt 
valid = /home/pdd/pdwork/darknet2/darknet/VOC/cat/validateImagePath.txt 
names = data/cats.names 
classes存放类别总数（这里只有cat一种），train 和valid 放着的是训练图片和验证图片的路径，cats.names存放的是方框注释，这里只有cat一行： 

yolo-voc.cfg 
将[region]中的classes改为1（这里只有cat一类），将最后一个[convolutional]（紧挨着[region]的前一个）中的filter改为30（filter的公式filters=(classes+ coords+ 1)* (NUM) ，我的是(1+4+1)* 5=30）。 
c.cats.names 
在data文件夹下新建cats.names，

4. 下载预训练文件cfg/darknet19_448.conv.23 
以在其他数据集上pretrain的模型做为初值，下载地址： 
链接：http://pan.baidu.com/s/1dFgUk4x 密码：ynhg。放在darknet文件夹下。

5. 训练 
在darknet文件夹路径下运行命令： 
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg cfg/darknet19_448.conv.23 
系统默认会迭代45000次batch，如果需要修改训练次数，进入cfg/yolo_voc.cfg修改max_batches的值。 
6. 测试 
训练完成后，模型成功保存，输入命令测试一下这个模型吧： 
./darknet detector test cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_final.weights testpicture/001.jpg



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#VOC训练
1. 下载voc数据集

wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

Annotations文件夹:图片的xml　xml主要为类别和坐标信息
ImageSets/Main:主要为train.txt test.txt val.txt 存放的是图片名
JPEGImages:图片


并将数据集放到data/voc中

2. 生成数集
wget https://pjreddie.com/media/files/voc_label.py
python voc_label.py

data/voc目录下有如下文件
2007_test.txt   VOCdevkit
2007_train.txt  voc_label.py
2007_val.txt    VOCtest_06-Nov-2007.tar
2012_train.txt  VOCtrainval_06-Nov-2007.tar
2012_val.txt    VOCtrainval_11-May-2012.tar

VOCdevkit/VOC2007/labels/ 和 VOCdevkit/VOC2012/labels/  目录下放的是labels

生成训练集:
    cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
3. 修改cfg/voc.data
   classes= 20   
   train  = <path-to-voc>/train.txt
   valid  = <path-to-voc>2007_test.txt
   names = data/voc.names
   backup = backup
  
  修改yolo-voc.cfg
  可修改batch ,max_batches  #batch 改为25，太大的化我的计算机带不动，出错
将[region]中的classes改为1（这里只有cat一类），将最后一个[convolutional]（紧挨着[region]的前一个）中的filter改为30（filter的公式filters=(classes+ coords+ 1)* (NUM) ，我的是(1+4+1)* 5=30）。125->30 
  解释下最后层filers:yolo2采用Anchor Boxes,最后层13x13xfilter
      classes:类别数
      coords:坐标
      1:概率
      NUM:anchor boxes数,这里采用5个即预测建议框,也可修改其他 
　　　　　一个boxes：[坐标＋框confidence＋异常每个类别的概率]=[x,y,w,h,0.75,0.45,0.78]
      

voc.names 改为当前类
c.cats.names 
4. 下载前训练完成模型（https://pjreddie.com/darknet/imagenet/#extraction）

wget https://pjreddie.com/media/files/darknet19_448.conv.23
或自己训练（https://pjreddie.com/darknet/imagenet/#darknet19_448）
./darknet partial cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23

5. 训练
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
如果需要限定gpu数量进行训练
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23 -gpus 0,1,2,3
从backup恢复训练
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_700.weights

训练完测试
./darknet detector test cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_final.weights testpicture/001.jpg
./darknet detector demo cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_final.weights  test.avi
############训练自己的数据集############
需要做成VOC数集一样
收集的图片格式为jpg格式,图片名最好是6位如 100333.jpg
先用图片生成train.txt(运行imgprocess.py),将train.txt中前30%或20%　剪切到test.txt 
VOCdevkit/VOCme
Annotations:放入xml文件
ImageSets/Main:train.txt test.txt val.txt　文件里为图片名字不含后缀。
JPEGImages:将图片拷贝进此文件夹　　图片格式jpg格式

修改运行voc_label.py:
	sets=[('me', 'train'), ('me', 'test')]
        classes = ["nan","nv"]


修改yolo-voc.cfg 
 	classes=2
    [region]生词一层卷积层filter改为35
修改voc.names
	nan
	nv
修改voc.data:类别数和数据路径
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#对于coco数集参考voc做部分修改

cp scripts/get_coco_dataset.sh data
cd data
bash get_coco_dataset.sh

修改coco.data
   classes= 80
   train  = <path-to-coco>/trainvalno5k.txt
   valid  = <path-to-coco>/5k.txt
   names = data/coco.names
   backup = backup

修改yolo.cfg
[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=25                       #64
subdivisions=8
....

训练
./darknet detector train cfg/coco.data cfg/yolo.cfg darknet19_448.conv.23
测试
./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg
./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights <video file>


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#训练CIFAR-10
1. 下载数集
cd data
wget https://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz

2. 生成路径
cd cifar
find `pwd`/train -name \*.png > train.list
find `pwd`/test -name \*.png > test.list
cd ../..

3. cfg/cifar.data

classes=10
train  = data/cifar/train.list
valid  = data/cifar/test.list
labels = data/cifar/labels.txt
backup = backup/
top=2   
#classes=10: the dataset has 10 different classes
#train = ...: where to find the list of training files
#valid = ...: where to find the list of validation files
#labels = ...: where to find the list of possible classes
#backup = ...: where to save backup weight files during training
#calculate top-n accuracy at test time (in addition to top-1)

4. cfg/cifar_small.cfg

5. train
./darknet classifier train cfg/cifar.data cfg/cifar_small.cfg










