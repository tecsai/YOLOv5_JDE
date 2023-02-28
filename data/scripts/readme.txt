以下只针对VOC数据(类型)集：
1. FilterImagesFromAnnos.py：筛选数据，当标注文件与图片数量不匹配时，删除多余的图片数据；
2. create_train_val_list.py：创建train，val，trainval，test数据列表(无路径，无后缀)
3. voc_convert.py：创建相应的文件夹，并处理VOC数据格式，生成yolo格式数据，存放于相应的文件夹中
   CAUTION: voc_convert中需要指明类别列表
4. voc_reorg.py：转存相应的图片数据,将相应图片复制到images(包含train和val)文件夹下