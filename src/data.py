# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import Literal, List, Dict
import os
import sys
import math
import logging
import functools
import braceexpand
import glob
import random
import pdb
import json

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000                                                                                              
import PIL
from typing import Union
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torchvision.datasets.folder import DatasetFolder
import torchvision.datasets as datasets
import torchvision.transforms as T
from third_party.open_clip.clip import tokenize

import webdataset as wds
import webdataset.handlers as wds_handlers
import webdataset.utils as wds_utils

DILATION = 0.7
PAD_CROP = True


def _seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class GeneCISDataset(Dataset):
    def __init__(self, data_root, json_root, task, transforms, tokenizer=None):
        super().__init__()
        self.transform = transforms
        self.tokenizer = tokenizer 
        self.task = task
        self.dilate = DILATION
        self.pad_crop = PAD_CROP
        
        self.vg_root = "/data2/mingyu/composed_image_retrieval/data/genecis/VG_100K"
        self.coco_root = "/data2/mingyu/composed_image_retrieval/data/coco/val2017"
        
        json_path = os.path.join(json_root, f'{task}.val.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(json_root, f'{task}.json')
            
        print(f"Loading GeneCIS data from {json_path}")
        with open(json_path, 'r') as f:
            self.val_samples = json.load(f)

    def load_cropped_image(self, img_meta):
        bbox = None
        path = ""

        if isinstance(img_meta, dict):
            bbox = img_meta.get('instance_bbox', img_meta.get('bbox'))
            
            if 'val_image_id' in img_meta:
                image_id = img_meta['val_image_id']
                path = os.path.join(self.coco_root, f'{int(image_id):012d}.jpg')
            else:
                image_id = img_meta.get('image_id', img_meta.get('id'))
                path = os.path.join(self.vg_root, f'{image_id}.jpg')
        else:
            image_id = img_meta
            if 'object' in self.task:
                path = os.path.join(self.coco_root, f'{int(image_id):012d}.jpg')
            else:
                path = os.path.join(self.vg_root, f'{image_id}.jpg')

        try:
            im = Image.open(path).convert('RGB')
        except Exception:
            im = Image.new('RGB', (224, 224), (0, 0, 0))

        if bbox is not None:
            im_width, im_height = im.size
            width = bbox[2]
            height = bbox[3]

            if self.dilate:
                orig_left, orig_top = bbox[0], bbox[1]
                left, top = max(0, orig_left - self.dilate * width), max(0, orig_top - self.dilate * height)
                right, bottom = min(im_width, left + (1 + self.dilate) * width), min(im_height, top + (1 + self.dilate) * height)
            else:
                left, top = bbox[0], bbox[1]
                right, bottom = bbox[0] + width, bbox[1] + height

            im = im.crop((left, top, right, bottom))

            if self.pad_crop:
                im = expand2square(im, (0, 0, 0))
        
        if self.transform is not None:
            im = self.transform(im)

        return im

    def __getitem__(self, index):
        sample = self.val_samples[index]
        
        reference_meta = sample['reference']
        target_meta = sample['target']
        gallery_metas = sample['gallery']
        caption = sample['condition']

        ref_img = self.load_cropped_image(reference_meta)
        target_img = self.load_cropped_image(target_meta)
        gallery_imgs = [self.load_cropped_image(g) for g in gallery_metas]

        all_gallery_imgs = [target_img] + gallery_imgs
        
        if self.transform is not None:
            gallery_tensor = torch.stack(all_gallery_imgs)
        
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return {
            'ref_img': ref_img,
            'caption': caption,
            'gallery_set': gallery_tensor,
            'target_rank': 0
        }

    def __len__(self):
        return len(self.val_samples)


class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'], mode: Literal['relative', 'classic'],transforms: callable,
                 preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path
        self.transforms = transforms

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def get_semantic_aspects(self, index):
        """ Returns the semantic aspects for a given query"""
        return self.annotations[index].get('semantic_aspects', [])

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")





## Structure of dataset directory
## CIRR: under ./data/CIRR
## validation images ./dev/
## caption split ./captions/cap.rc2.val.json
## image split ./image_splits/split.rc2.val.json
class CIRR(Dataset):
    def __init__(self, transforms, mode='caps', 
    vis_mode=False, test=False, root='./data'):
        self.mode = mode
        self.transforms = transforms
        self.vis_mode = vis_mode
        ## mode to use test split of CIRR
        self.test = test
        self.root = os.path.join(root, 'CIRR')
        self.root_img = os.path.join(self.root, 'dev')
        if self.test:
            self.root_img = os.path.join(self.root, 'test1')
            if self.mode == 'caps':
                self.json = os.path.join(self.root , 'captions/cap.rc2.test1.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.test1.json')
        else:
            if self.mode == 'caps':
                self.json = os.path.join(self.root, 'captions/cap.rc2.val.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.val.json')
        logging.debug(f'Loading json data from {self.json}.')
        data = json.load(open(self.json, "r"))                                
        self.ref_imgs = []
        self.target_imgs = []
        self.target_caps = []        
        if self.test:
            self.init_test(data)
        elif self.mode == 'caps':            
            self.init_val(data)                        
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]                    
        if self.vis_mode:
            self.target_imgs = list(set(self.target_imgs))
        logging.info("Use {} imgs".format(len(self.target_imgs)))        

    def init_test(self, data):
        self.pairids = []
        if self.mode == 'caps':
            for d in data:
                ref_path = d['reference']+ ".png"
                self.ref_imgs.append(ref_path)
                self.target_caps.append(d['caption']) 
                self.pairids.append(d['pairid'])
                self.target_imgs.append('dummy')
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]

    def init_val(self, data):
        for d in data:
            ref_path = d['reference']+ ".png"
            tar_path = d['target_hard']+ ".png"
            self.ref_imgs.append(ref_path)
            self.target_imgs.append(tar_path)
            self.target_caps.append(d['caption'])            
    
    def return_testdata(self, idx):
        if self.mode == 'caps':
                ref_path = str(self.ref_imgs[idx])
                img_path = os.path.join(self.root_img, ref_path)
                ref_images = self.transforms(Image.open(img_path))
                target_cap = self.target_caps[idx]
                text_with_blank_raw = 'a photo of * and {}'.format(target_cap)    
                caption_only = tokenize(target_cap)[0]
                text_with_blank = tokenize(text_with_blank_raw)[0]                 
                return ref_images, text_with_blank, \
                    caption_only, str(self.ref_imgs[idx]), \
                        self.pairids[idx], text_with_blank_raw
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = Image.open(os.path.join(self.root_img, tar_path))
            target_images = self.transforms(img_path)
            return target_images, tar_path

    def return_valdata(self, idx):
        if self.mode == 'caps' and not self.vis_mode:
            ref_path = str(self.ref_imgs[idx])
            img_path = os.path.join(self.root_img, ref_path)
            ref_images = self.transforms(Image.open(img_path))
            target_cap = self.target_caps[idx]
            text_with_blank = 'a photo of * and {}'.format(target_cap) 
            caption_only = tokenize(target_cap)[0]
            ref_text_tokens = tokenize(text_with_blank)[0]                 
            return ref_images, ref_text_tokens, caption_only, \
                str(self.ref_imgs[idx]), str(self.target_imgs[idx]), \
                    target_cap                       
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = os.path.join(self.root_img, tar_path)
            target_images = self.transforms(Image.open(img_path))
            return target_images, img_path

    def __getitem__(self, idx):
        if self.test:                        
            return self.return_testdata(idx)
        else:
            return self.return_valdata(idx)
    
    def __len__(self):
        return len(self.target_imgs)
        
## Fashion-IQ: under ./data/fashion-iq
## validation images ./images
## caption split ./json/cap.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
## image split ./image_splits/split.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
class FashionIQ(Dataset):
    def __init__(self, cloth, transforms, is_train=False, vis_mode=False, \
        mode='caps', is_return_target_path=False, root='./data'):
        root_iq = os.path.join(root, 'fashion-iq')
        self.root_img = os.path.join(root_iq, 'images')
        self.vis_mode = vis_mode
        self.mode = mode
        self.is_return_target_path = is_return_target_path
        self.transforms = transforms
        if mode == 'imgs':
            self.json_file = os.path.join(root_iq, 'image_splits', \
                'split.{}.val.json'.format(cloth))
        else:
            self.json_file = os.path.join(root_iq, 'json', \
                'cap.{}.val.json'.format(cloth))                
        logging.debug(f'Loading json data from {self.json_file}.')

        self.ref_imgs = []
        self.target_imgs = []
        self.ref_caps = []
        self.target_caps = []        
        if mode == 'imgs':
            self.init_imgs()
            logging.info("Use {} imgs".format(len(self.target_imgs)))
        else:
            self.init_data()     
            logging.info("Use {} imgs".format(len(self.target_imgs)))

    def init_imgs(self):
        data = json.load(open(self.json_file, "r"))
        self.target_imgs = [key + ".png" for key in data]        

    def init_data(self):
        def load_data(data):
            for d in data:
                ref_path = os.path.join(self.root_img, d['candidate']+ ".png") 
                tar_path = os.path.join(self.root_img, d['target']+ ".png")            
                try:
                    Image.open(ref_path)
                    Image.open(tar_path)
                    self.ref_imgs.append(ref_path)
                    self.target_imgs.append(tar_path)
                    self.ref_caps.append((d['captions'][0], d['captions'][1]))
                    #self.target_caps.append(d['captions'][1])
                except:                
                    print('cannot load {}'.format(d['candidate']))
        if isinstance(self.json_file, str):
            data = json.load(open(self.json_file, "r"))        
            load_data(data)            
        elif isinstance(self.json_file, list):
            for filename in self.json_file:
                data = json.load(open(filename, "r")) 
                load_data(data)         

    def __len__(self):
        if self.mode == 'caps':
            return len(self.ref_imgs)
        else:
            return len(self.target_imgs)

    def return_imgs(self, idx):
        tar_path = str(self.target_imgs[idx])
        img_path = os.path.join(self.root_img, tar_path)
        target_images = self.transforms(Image.open(img_path))
        return target_images, os.path.join(self.root_img, tar_path)

    def return_all(self, idx):
        if self.vis_mode:
            tar_path = str(self.target_imgs[idx])
            target_images = self.transforms(Image.open(tar_path))
            return target_images, tar_path            
        ref_images = self.transforms(Image.open(str(self.ref_imgs[idx])))
        target_images = self.transforms(Image.open(str(self.target_imgs[idx])))
        cap1, cap2 = self.ref_caps[idx]
        text_with_blank = 'a photo of * and {} and {}'.format(cap2, cap1)
        token_texts = tokenize(text_with_blank)[0]                
        if self.is_return_target_path:
            return ref_images, target_images, token_texts, token_texts, \
                str(self.target_imgs[idx]), str(self.ref_imgs[idx]), \
                    (cap1, cap2)
        else:
            return ref_images, target_images, text_with_blank


    def __getitem__(self, idx):
        if self.mode == 'imgs':            
            return self.return_imgs(idx)
        else:            
            return self.return_all(idx)
        
## COCO: under ./data/coco
## validation images ./val2017
## validation masked images ./val2017_masked
## validation csv file ./coco_eval.csv
class CsvCOCO(Dataset):
    def __init__(self, transforms, transforms_region, sep=",",
                return_data_identifier=False, return_filename=False, 
                root='./data'):
        self.transforms = transforms
        self.transforms_region = transforms_region
        self.root = os.path.join(root, 'coco')
        self.root_img = os.path.join(self.root, 'val2017')
        self.csv_file = os.path.join(self.root, 'coco_eval.csv')
        logging.debug(f'Loading csv data from {self.csv_file}.')
        df = pd.read_csv(self.csv_file, sep=sep)                
        self.images = df['id'].tolist()
        ## query_region contains the box of query regions.
        regions = df['query_regions'].tolist()
        self.regions = []
        for region in regions:
            x1, y1, x2, y2 = map(lambda x: int(float(x)), region.split(";"))
            self.regions.append([x1, y1, x2, y2])

        ## query_classes contains the class of query region in the target.
        self.query_classes = df['query_class'].tolist()
        self.classes = []
        ## classes contains the list of classes in the target.
        for list_class in df['classes'].tolist():
            if isinstance(list_class, str):
                list_class = list_class.split(";")
                self.classes.append(list_class)
            else:
                self.classes.append([""])        
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_img, str(self.images[idx]))
        image = Image.open(img_path)        
        masked_path = os.path.join(self.root_img.replace('val2017', 'val2017_masked'), \
            str(self.images[idx]))
        image_masked = Image.open(masked_path)
        
        ## extract query region.
        x1, y1, x2, y2 = self.regions[idx]        
        region_image = image_masked.crop((x1, y1, x2, y2)) 

        image = self.transforms(image)
        ## no cropping is applied to query region.
        region_image = self.transforms_region(region_image)
        query_class = self.query_classes[idx]
        other_classes = self.classes[idx]        
        text_with_blank = 'a photo of * and {}'.format(" and ".join(other_classes))
        text_with_queryclass = 'a photo of * and {} and {}'.format(query_class, \
            " and ".join(other_classes))
        raw_text = text_with_queryclass
        text_full = 'a photo of {} and {}'.format(query_class, " and ".join(other_classes))        
        text_with_blank = tokenize(text_with_blank)[0]
        text_with_queryclass = tokenize(text_with_queryclass)[0]
        text_full = tokenize(text_full)[0]
        return image, region_image, text_full, text_with_blank, \
            text_with_queryclass, str(self.images[idx]), raw_text


class ImageList(Dataset):
    def __init__(self, input_filename, transforms, root=None, 
                 return_filename=False, is_labels=False):
        logging.debug(f'Loading txt data from {input_filename}.')
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        if not is_labels:
            self.images = [line.strip() for line in lines]
        else:
            filenames = [line.strip() for line in lines]
            self.images = [name.split(" ")[0] for name in filenames] 
            self.labels = [int(name.split(" ")[1]) for name in filenames] 
        self.is_labels = is_labels
        self.transforms = transforms
        self.root = root
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root is not None:
            img_path = os.path.join(self.root, str(self.images[idx]))
        else:
            img_path = str(self.images[idx])
        images = self.transforms(Image.open(img_path))
        if self.return_filename:
            return images, img_path
        elif self.is_labels:
            target = self.labels[idx]
            return images, target       
        else:
            return images


class CustomFolder(Dataset):
    def __init__(self, folder, transform):
        image_lists = os.listdir(folder)
        self.samples = [os.path.join(folder, name) for name in image_lists]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = Image.open(str(path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t",
                 return_data_identifier=False, return_filename=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.return_filename:
            return images, str(self.images[idx])
        texts = tokenize([str(self.captions[idx])])[0]

        if self.return_data_identifier:
            return images, texts, 0
        return images, texts

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def _load_cc3m_cir_jsonl(jsonl_path: str, reverse_jsonl_path: str = None):
    """id -> (instruction, modified_caption, reverse_instruction)"""
    
    def _normalize_instruction(x):
        """
        Extract instruction text. Can look for instruction, text, value, en keys.
        """
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            # For instruction, we can look for instruction, text, value, en
            # But NEVER include "brainstorming" or "modified_caption"
            for k in ["instruction", "text", "value", "en"]:
                if k in x and isinstance(x[k], str):
                    return x[k]
            return ""
        if isinstance(x, (list, tuple)):
            # First, try to find a string element
            for v in x:
                if isinstance(v, str):
                    return v
            # Second, try to find a dict with instruction key
            for v in x:
                if isinstance(v, dict):
                    extracted = _normalize_instruction(v)
                    if extracted:
                        return extracted
            return ""
        return str(x)
    
    def _normalize_modified_caption(x):
        """
        Extract modified_caption text with smart fallback.
        CRITICAL: Prioritize "modified_caption" key, NEVER include "brainstorming" or "instruction" content.
        This prevents token truncation issues (CLIP max 77 tokens) where brainstorming content
        could push out the actual caption text.
        
        Fallback strategy (in order):
        1. "modified_caption" key (highest priority)
        2. "caption" key (common alias)
        3. Other text keys (text, value, en) - but EXCLUDE "brainstorming" and "instruction"
        4. First string element in list
        5. Empty string (last resort)
        """
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            # Priority 1: "modified_caption" key (highest priority)
            if "modified_caption" in x and isinstance(x["modified_caption"], str):
                return x["modified_caption"]
            # Priority 2: "caption" key (common alias)
            if "caption" in x and isinstance(x["caption"], str):
                return x["caption"]
            # Priority 3: Other text keys, but EXCLUDE "brainstorming" and "instruction"
            # This provides fallback for data format variations while avoiding contamination
            excluded_keys = {"brainstorming", "instruction"}  # NEVER include these
            for k in ["text", "value", "en", "description", "content"]:
                if k in x and k not in excluded_keys and isinstance(x[k], str):
                    return x[k]
            # Last resort: return empty (don't stringify entire dict to avoid contamination)
            return ""
        if isinstance(x, (list, tuple)):
            # Priority 1: Try to find a dict with "modified_caption" key
            for v in x:
                if isinstance(v, dict):
                    extracted = _normalize_modified_caption(v)
                    if extracted:  # Only return if we found something
                        return extracted
            # Priority 2: Try to find first string element (assume it's the caption)
            # This handles cases like ["caption text", {...}] or ["caption text"]
            for v in x:
                if isinstance(v, str):
                    return v
            # Last resort: return empty string (don't join all elements to avoid contamination)
            return ""
        return str(x)

    def _normalize_reverse_instruction(x):
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ["reverse_instruction", "instruction", "text", "value", "en"]:
                if k in x and isinstance(x[k], str):
                    return x[k]
            return ""
        if isinstance(x, (list, tuple)):
            for v in x:
                if isinstance(v, str):
                    return v
            for v in x:
                if isinstance(v, dict):
                    extracted = _normalize_reverse_instruction(v)
                    if extracted:
                        return extracted
            return ""
        return str(x)

    mp = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            _id = str(obj.get("id"))
            ins = _normalize_instruction(obj.get("instruction", ""))
            cap = _normalize_modified_caption(obj.get("modified_caption", ""))
            rev = _normalize_reverse_instruction(obj.get("reverse_instruction", ""))
            mp[_id] = (ins, cap, rev)

    if reverse_jsonl_path and os.path.exists(reverse_jsonl_path):
        updated = 0
        extra = 0
        with open(reverse_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                _id = str(obj.get("id"))
                rev = _normalize_reverse_instruction(obj.get("reverse_instruction", ""))
                if not rev:
                    continue
                if _id not in mp:
                    extra += 1
                    continue
                ins, cap, _ = mp[_id]
                mp[_id] = (ins, cap, rev)
                updated += 1
        logging.info(
            f"Loaded reverse sidecar from {reverse_jsonl_path}: updated={updated} extra_ids_ignored={extra}"
        )
    return mp


def _cc3m_cir_wds_select(sample: dict, id2meta: dict) -> bool:
    """Top-level function so it can be pickled under multiprocessing spawn."""
    k = sample.get("__key__", None)
    return (k is not None) and (k in id2meta)


def _cc3m_cir_wds_attach(sample: dict, id2meta: dict) -> dict:
    """Attach instruction/modified_caption/reverse_instruction to decoded sample."""
    k = sample["__key__"]
    ins, cap, rev = id2meta[k]
    src_caption = sample.get("src_caption", "")
    if isinstance(src_caption, bytes):
        src_caption = src_caption.decode("utf-8", errors="ignore")
    elif src_caption is None:
        src_caption = ""
    else:
        src_caption = str(src_caption)
    return {
        "id": k,
        "ref_img": sample["image"],  # PIL after decode
        "src_caption": src_caption,
        "instruction": ins,
        "modified_caption": cap,
        "reverse_instruction": rev,
    }


def _cc3m_cir_wds_to_tensor(sample: dict, preprocess_fn) -> dict:
    """Convert PIL/numpy image to tensor using preprocess_fn."""
    img = sample["ref_img"]
    # 容错处理: 有时 decode 可能会返回 numpy array
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img).convert("RGB")
    else:
        img = img.convert("RGB")
    sample["ref_img"] = preprocess_fn(img)
    return sample


def _cache_shards(urls, cache_dir):
    """
    Cache remote shard URLs to local directory.
    Returns a generator that yields local paths (or original URLs if caching fails).
    """
    import urllib.request
    import hashlib
    from pathlib import Path
    
    os.makedirs(cache_dir, exist_ok=True)
    
    for url in urls:
        if url.startswith(('http://', 'https://')):
            # Generate cache filename from URL
            url_hash = hashlib.md5(url.encode()).hexdigest()
            url_filename = os.path.basename(url.split('?')[0])  # Remove query params
            cache_path = os.path.join(cache_dir, f"{url_hash}_{url_filename}")
            
            # Check if already cached
            if os.path.exists(cache_path):
                yield cache_path
            else:
                # Download to cache
                try:
                    logging.info(f"Downloading {url} to {cache_path}")
                    urllib.request.urlretrieve(url, cache_path)
                    yield cache_path
                except Exception as e:
                    logging.warning(f"Failed to cache {url}: {e}, using original URL")
                    yield url
        else:
            # Local path, return as-is
            yield url


def get_cc3m_cir_wds(args, preprocess_fn, is_train, input_filename=None):
    """
    完全兼容模式：将所有处理步骤放入 pipeline 列表构建
    """
    assert args.cc3m_cir_jsonl is not None, "--cc3m-cir-jsonl is required"
    assert args.wds_shards is not None, "--wds-shards is required"

    # 加载元数据
    id2meta = _load_cc3m_cir_jsonl(
        args.cc3m_cir_jsonl,
        reverse_jsonl_path=getattr(args, "cc3m_cir_reverse_jsonl", None),
    )
    exts = args.wds_image_key.split(";")
    rename_str = ";".join(exts)

    # NOTE: use top-level functions + partial for multiprocessing spawn picklability
    select_fn = functools.partial(_cc3m_cir_wds_select, id2meta=id2meta)
    attach_fn = functools.partial(_cc3m_cir_wds_attach, id2meta=id2meta)
    to_tensor_fn = functools.partial(_cc3m_cir_wds_to_tensor, preprocess_fn=preprocess_fn)
    deterministic_wds = bool(is_train and getattr(args, "wds_deterministic", False))
    base_seed = int(getattr(args, "seed", 0))
    if deterministic_wds:
        logging.info(f"Using deterministic WebDataset sampling with base_seed={base_seed}")

    # --- 2. 构建 Pipeline 节点列表 ---
    pipeline = []

    # [预处理 shards] 展开 glob pattern 和 brace expansion，处理缓存
    shards_pattern = args.wds_shards
    if isinstance(shards_pattern, str):
        if shards_pattern.count("{") != shards_pattern.count("}"):
            raise ValueError(
                f"Malformed --wds-shards pattern: unmatched braces in {shards_pattern!r}"
            )
    
    # 检查是否是 glob pattern（包含 * 或 ?）
    if '*' in shards_pattern or '?' in shards_pattern or '{' in shards_pattern:
        # 先展开 brace expansion（如 {0000..0575}）
        expanded = list(braceexpand.braceexpand(shards_pattern))
        # 再展开 glob pattern
        shard_list = []
        for pattern in expanded:
            if '*' in pattern or '?' in pattern:
                shard_list.extend(glob.glob(pattern))
            else:
                shard_list.append(pattern)
        shards_pattern = shard_list
        if len(shards_pattern) == 0:
            raise FileNotFoundError(
                f"--wds-shards expanded to 0 files. Original pattern: {args.wds_shards!r}"
            )
        logging.info(f"Expanded shards pattern to {len(shards_pattern)} files")
    
    # [缓存预处理] 如果指定了缓存目录且 shards 是远程 URL，先缓存到本地
    if getattr(args, "wds_cache_dir", None) is not None:
        cache_dir = args.wds_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Using WebDataset cache directory: {cache_dir}")
        
        # 如果 shards_pattern 是字符串（未展开），先展开
        if isinstance(shards_pattern, str):
            shard_urls = list(braceexpand.braceexpand(shards_pattern))
        else:
            shard_urls = shards_pattern
        
        # 只缓存远程 URL
        cached_shards = []
        for url in shard_urls:
            if isinstance(url, str) and url.startswith(('http://', 'https://')):
                cached_shards.extend(_cache_shards([url], cache_dir))
            else:
                cached_shards.append(url)
        
        shards_pattern = cached_shards

    # [源头]
    if is_train and args.wds_resampled:
        if deterministic_wds:
            pipeline.append(
                wds.ResampledShards(
                    shards_pattern,
                    seed=base_seed,
                    worker_seed=wds_utils.pytorch_worker_seed,
                    deterministic=True,
                )
            )
        else:
            pipeline.append(wds.ResampledShards(shards_pattern))
    else:
        pipeline.append(wds.SimpleShardList(shards_pattern))

    # [DDP 分片]
    if args.distributed and is_train:
        pipeline.append(wds.split_by_node)

    # [Worker 分片]
    pipeline.append(wds.split_by_worker)

    # [Shard 级 Shuffle]
    if is_train and not args.wds_resampled:
        pipeline.append(
            wds.shuffle(
                args.wds_shardshuffle,
                seed=(base_seed + 17) if deterministic_wds else None,
            )
        )

    # [解包]
    # Some cached shards may be partially downloaded/corrupted; skip bad tar entries instead of crashing.
    pipeline.append(wds.tarfile_to_samples(handler=wds_handlers.warn_and_continue))

    # [样本级 Shuffle]
    if is_train:
        pipeline.append(
            wds.shuffle(
                args.wds_shuffle,
                seed=(base_seed + 23) if deterministic_wds else None,
            )
        )

    # [解码] 使用 wds.decode 函数
    pipeline.append(wds.decode("pil"))

    # [重命名] 使用 wds.rename 函数
    text_key = getattr(args, "wds_text_key", "txt;text;caption")
    pipeline.append(wds.rename(image=rename_str, src_caption=text_key, __key__="__key__"))

    # [过滤 & 映射] 使用 wds.select / wds.map 函数
    pipeline.append(wds.select(select_fn))
    pipeline.append(wds.map(attach_fn))
    pipeline.append(wds.map(to_tensor_fn))

    # --- 3. 创建 DataPipeline 对象 ---
    # 过滤掉可能存在的 None
    pipeline = [s for s in pipeline if s is not None]
    ds = wds.DataPipeline(*pipeline)

    # --- 4. DataLoader ---
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        worker_init_fn=_seed_dataloader_worker if is_train else None,
    )

    # 标记为流式数据 (长度未知)
    dataloader.num_samples = None
    dataloader.num_batches = None
    
    return DataInfo(dataloader, None)
    
def preprocess_txt(text):
    return tokenize([str(text)])[0]

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    sizes = json.load(open(sizes_filename, 'r'))
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path  = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )
    return DataInfo(dataloader, sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def get_csv_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


#
def get_imgnet_r(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    path_data = os.path.join(args.root_data, 'imgnet/imagenet-r')
    dataset = CustomFolder(path_data, transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)


def get_directory_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomFolder(
        input_filename,
         transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_circo(args, preprocess_fn, is_train, input_filename=None):
    # CIRCO 主要是为了评估 (val/test)，通常不用作训练集，除非你是做伪标签训练
    # 这里假设我们用它做 zero-shot 评估
    split = 'train' if is_train else 'val' 
    # 如果 args.test 为真，则加载 test split
    if hasattr(args, 'test') and args.test:
        split = 'test'
        
    dataset = CIRCODataset(transforms=preprocess_fn, split=split, root=args.root_data)
    
    # DataLoader 配置...
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # 评估时不打乱
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return DataInfo(dataloader, None)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'imgnet_r':
        return get_imgnet_r
    elif dataset_type == 'fashion-iq':
        return get_fashion_iq
    elif dataset_type == 'cirr':
        return get_cirr
    elif dataset_type == 'directory':
        return get_directory_dataset
    elif dataset_type == "csv":
        return get_csv_dataset     
    elif dataset_type == 'circo':
        return get_circo   
    elif dataset_type == "cc3m_cir_wds":
        return get_cc3m_cir_wds
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    dataset_type_val = getattr(args, 'dataset_type_val', args.dataset_type)
    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type_val)(
            args, preprocess_val, is_train=False)
    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")
    return data
