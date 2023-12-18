'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        # Where 
        self.data_root = cfg['data_root']
        self.split = split
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])
        
        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            cfg["annotate_root"],
            'train.csv' if self.split=='train' else 'valid.csv'
        )
        trainPath = os.path.join(
            os.path.dirname(annoPath),
            'train.csv'
        )
        meta = pd.read_csv(annoPath)
        train  = pd.read_csv(trainPath)      
        
        class_labels = cfg['class_labels']
        Y_train = train[class_labels]
        encoder = LabelEncoder()
        encoder.fit(Y_train)
        
        Y = meta[class_labels]
        labelIndex = encoder.transform(Y)
        
        file_name = cfg['file_name']
        imgFileName = meta[file_name]
        
        self.data = list(zip(imgFileName.tolist(), labelIndex.tolist()))

        # images = dict([[i['id'], i['file_name']] for i in meta['images']])          # image id to filename lookup
        # labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])]) # custom labelclass indices that start at zero
        
        # # since we're doing classification, we're just taking the first annotation per image and drop the rest
        # images_covered = set()      # all those images for which we have already assigned a label
        # for anno in meta['annotations']:
        #     imgID = anno['image_id']
        #     if imgID in images_covered:
        #         continue
            
        #     # append image-label tuple to data
        #     imgFileName = images[imgID]
        #     label = anno['category_id']
        #     labelIndex = labels[label]
        #     self.data.append([imgFileName, labelIndex])
        #     images_covered.add(imgID)       # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root, 'AllPhotosJPG', image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label
