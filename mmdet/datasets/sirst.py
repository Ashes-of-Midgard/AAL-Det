import os
import os.path as osp
import xml.etree.ElementTree as ET

from .base_det_dataset import BaseDetDataset
from ..registry import DATASETS

@DATASETS.register_module()
class SirstDataset(BaseDetDataset):
    METAINFO = {
        'classes': ('Target',)
    }

    def __init__(self,
                 *args,
                 split: str='train_full',
                 **kwargs):
        self.split= split
        super().__init__(*args,
                         **kwargs)

    def load_data_list(self):
        '''
            - img_path：图像的文件路径，用于定位和读取具体的图像数据。
            - img_shape：图像的形状，通常表示为 (height, width, channels)，用于确定图像的尺寸和通道数，以便在数据预处理和模型输入时进行必要的调整。
            - instances：是一个列表，其中每个元素通常是一个字典，包含单个实例的标注信息，常见的键值对有：
                - bbox：表示边界框的坐标，通常格式为 [x1, y1, x2, y2]，分别对应边界框左上角和右下角的坐标。
                - bbox_label：边界框对应的类别标签，通常是一个整数索引，与类别列表中的类别相对应。
                - mask：如果存在实例分割任务，该键对应的值为实例的掩码数据，通常是一个二维或三维的数组，用于表示实例在图像中的具体区域。
                - segm：语义分割的标注信息，可能是多边形坐标表示的分割区域或其他形式的分割标注。
        '''
        img_root = osp.join(self.data_root, 'mixed')
        anno_root = osp.join(self.data_root, 'annotations', 'bboxes')
        data_list = []
        with open(osp.join(self.data_root, 'splits', self.split+'.txt'), 'r') as split_file:
            file_names = split_file.readlines()
            for name in file_names:
                name = name.split('\n')[0]
                xml_path = osp.join(anno_root, name+'.xml')
                img_path = osp.join(img_root, name+'.png')
                tree = ET.parse(xml_path)
                et_root = tree.getroot()
                size = et_root.find('size')
                img_shape = (int(size.findtext('height')), int(size.findtext('width')), int(size.findtext('depth')))
                instances = []
                objects = et_root.findall('object')
                for obj in objects:
                    bbox = obj.find('bndbox')
                    if bbox is not None:
                        xmin = int(bbox.findtext('xmin'))
                        ymin = int(bbox.findtext('ymin'))
                        xmax = int(bbox.findtext('xmax'))
                        ymax = int(bbox.findtext('ymax'))
                        bbox = (xmin, ymin, xmax, ymax)
                        bbox_label = 0
                        instances.append({
                            'bbox': bbox,
                            'bbox_label': bbox_label,
                            'ignore_flag': 0
                        })
                data_list.append({
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'instances': instances
                })
        return data_list
