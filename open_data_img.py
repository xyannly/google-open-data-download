# @File  : open_data_img.py
# @Author: Ann
# @Date  : 2020/1/6
import pandas as pd
import os
import cv2
import numpy as np
import sys
import re
import gc
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import datetime
import socket

path = 'F:\\目标检测\\Google open\\'
os.chdir(path)
LabelName_csv = 'class-descriptions-boxable.csv'
ImageId_csv = 'train-annotations-bbox.csv'
ImageUrl_csv = 'train-images-boxable-with-rotation.csv'
kwargs = {'header': None, 'names': ['LabelID', 'LabelName']}
class_names = pd.read_csv(LabelName_csv, **kwargs)
train_boxed = pd.read_csv(ImageId_csv,usecols=['ImageID','LabelName'])
image_ids = pd.read_csv(ImageUrl_csv,usecols=['ImageID','OriginalURL'])

def count_time(func):
    def int_time(*args, **kwargs):
        start_time = datetime.datetime.now()  # 程序开始时间
        func(*args, **kwargs)
        over_time = datetime.datetime.now()   # 程序结束时间
        total_time = (over_time-start_time).total_seconds()
        print("download picture used time %s seconds" % total_time)
    return int_time

def callbackinfo(down, block, size):
    '''
    回调函数：
    down：已经下载的数据块
    block：数据块的大小
    size：远程文件的大小
    '''
    per = 100.0 * (down * block) / size
    if per > 100:
        per = 100
    print('%.2f%%' % per)

@count_time
def download(id,url,out_path):
    filename ='{}.jpg'.format(id)
    if filename in os.listdir(out_path):
        return 0
    try:
        print('start download '+filename)
        urllib.request.urlretrieve(url,out_path +'/'+ filename,callbackinfo)

        #request = urllib.request.Request(url)
        #response = urllib.request.urlopen(request)
        #get_img = response.read()
        #with open(out_path +'/'+ filename, 'wb') as f:
        #    f.write(get_img)
        print(filename + ' download successfully!')
    except socket.timeout as ex:
        print(ex)
    except:
        print('download image error:  '+ filename)

    for x in locals().keys():
        del locals()[x]
    gc.collect()
    print('clear variables!!!')
def get_Images(input_imgname,class_names,train_boxed,image_ids):
    Label_map = dict(class_names.set_index('LabelName').loc[input_imgname, 'LabelID']
                     .to_frame().reset_index().set_index('LabelID')['LabelName'])
    #label_values = set(Label_map.keys())
    #Label_map = dict(map(lambda x,y:[x,y],input_labelid,input_imgname))
    print(Label_map)

    for id,name in Label_map.items():
        print(name)
        relevant_training_images = train_boxed[train_boxed.LabelName==id]
        #print(relevant_training_images)
        relevant_flickr_urls = (relevant_training_images.set_index('ImageID')
                               .join(image_ids.set_index('ImageID'))
                                .loc[:, 'OriginalURL'])
        #print(relevant_flickr_urls)
        del relevant_training_images
        gc.collect()
        out_path = os.path.join(path,name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        Image_id = []
        for index in relevant_flickr_urls.keys():
            if index in Image_id:
                continue
            Image_id.append(index)
            if len(Image_id)>1000:
                break
            url = relevant_flickr_urls.loc[index]
            socket.setdefaulttimeout(20)
            download(index,url,out_path)
            #print(index+'.jpg download successfully!')

def indent_add(elem, level=0):#递归方式解决
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():#如果字符串节点为空
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():#elem.tail表示该节点到下一个节点之间的内容
            elem.tail = i
        for elem in elem:
            indent_add(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def get_xmls(class_names,train_boxed):

    demo_file = 'F:\\目标检测\\Google open\\demo.xml'
    for root,dir,files in os.walk(path,topdown=False):
        os.chdir(root)
        if root == path:
            break
        label_name = root.split('\\')[-1]
        for file in files:
            if not re.search('.*jpg',file):
                continue
            img_id = file.split('.')[0]
            xml_file = img_id+'.xml'
            shutil.copy(demo_file,xml_file)
            tree = ET.parse(xml_file)
            rot = tree.getroot()
            rot.find('folder').text = label_name
            rot.find('filename').text = file
            rot.find('path').text = root+'\\'+file

            img = cv2.imdecode(np.fromfile(file,dtype=np.uint8),-1)
            width = img.shape[1]
            height = img.shape[0]
            depth = img.shape[2]
            rot.find('size/width').text = str(width)
            rot.find('size/height').text = str(height)
            rot.find('size/depth').text = str(depth)



            Label_id = class_names.set_index('LabelName').loc[label_name, 'LabelID']
            train_min = train_boxed[(train_boxed.LabelName==Label_id)&(train_boxed.ImageID==img_id)]
            print(train_min)

            for item in train_min.iterrows():
                xmin = int(item[1]['XMin']*width)
                ymin = int(item[1]['YMin']*height)
                xmax = int(item[1]['XMax']*width)
                ymax = int(item[1]['YMax']*height)
                if item[1]['IsOccluded']>0:
                    difficult = 1
                else:
                    difficult = 0
                if item[1]['IsTruncated'] > 0:
                    trun = 1
                else:
                    trun = 0
                object = ET.SubElement(rot, "object")#创建rot的子节点
                namen = ET.SubElement(object, "name")
                namen.text = label_name
                posen = ET.SubElement(object, "pose")
                posen.text = 'Unspecified'
                truncatedn = ET.SubElement(object, "truncated")
                truncatedn.text = str(trun)
                diff = ET.SubElement(object, "difficult")
                diff.text = str(difficult)
                bndboxn = ET.SubElement(object, "bndbox")
                Xmin = ET.SubElement(bndboxn, "xmin")
                Xmin.text = str(xmin)
                Ymin = ET.SubElement(bndboxn, "ymin")
                Ymin.text = str(ymin)
                Xmax = ET.SubElement(bndboxn, "xmax")
                Xmax.text = str(xmax)
                Ymax = ET.SubElement(bndboxn, "ymax")
                Ymax.text = str(ymax)
                indent_add(rot)
                tree.write(xml_file)


input_imgname = ['Cabbage','Footwear','Flower','Flashlight','Envelope','Doll','Drinking straw','Pitcher','Balance beam','Roller skates','Apple','Belt','Briefcase','Cabinetry','Coffee table','Countertop','Curtain']
#download('4949179909_5f4459df14_o','https://farm8.staticflickr.com/4113/4949179909_5f4459df14_o.jpg','./')#['Briefcase','Cabbage']sys.argv[1:]
#input_labelid = ['/m/044r5d','/m/0wdt60w','/m/0138tl','/m/06k2mb','/m/03ssj5']
get_Images(input_imgname,class_names,train_boxed,image_ids)
#get_xmls(class_names,train_boxed)
print(input_imgname)
