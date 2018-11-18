import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_loader_val(transform,
                   vocab_file='./vocab.pkl',
                   num_workers=0,
                   cocoapi_loc='/opt'):
    """Returns the validation data loader."""
    assert os.path.exists(vocab_file), "vocab_file does not exist."
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')
    
    # COCO caption dataset.
    dataset = CoCoValDataset(transform=transform,
                             vocab_file=vocab_file,
                             annotations_file=annotations_file,
                             img_folder=img_folder)

    return data.DataLoader(dataset=dataset,
                           batch_size=1,
                           shuffle=True,
                           num_workers=num_workers)

class CoCoValDataset(data.Dataset):
    def __init__(self, transform, vocab_file, annotations_file, img_folder):
        self.transform = transform
        self.vocab = Vocabulary(None, vocab_file, None, None, None, annotations_file, True)
        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())
        self.img_folder = img_folder
        test_info = json.loads(open(annotations_file).read())
        self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        path = self.paths[index]
        ann_id = self.ids[index]
        
        # Convert image to tensor and pre-process using transform
        PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(PIL_image)

        return image, self.coco.anns[ann_id]['image_id']

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.paths)