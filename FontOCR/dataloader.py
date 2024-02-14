import string
import os
import random
import torch
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image, ImageOps
from torch.utils import data

from ifontDataAdopter import IFontDataAdopter
from attr2font_data_adopter import Attr2FontDataAdopter
from large_scale_tag_based_font_data_adopter import LargeScaleTagBasedFontDataAdopter


only_lower = True
if (only_lower):
  chars = string.ascii_lowercase
else:
  chars = string.ascii_lowercase + string.ascii_uppercase

class FontOcrDataSet(data.Dataset):
  def __init__(self, dataset_root_dir, image_dir, font_names_file_name, attr2font_dataset, ignore_fonts_file_path, max_fonts_num = 10000):
    self.dataset_root_dir = dataset_root_dir
    self.image_dir = os.path.join(dataset_root_dir, image_dir)
    print('load image from', self.image_dir)
    self.font_names_path = os.path.join(dataset_root_dir, font_names_file_name)
    print('load font_names from', self.font_names_path)
    self.ignore_fonts_file_path = os.path.join(dataset_root_dir, ignore_fonts_file_path)
    self.max_fonts_num = max_fonts_num
    self.attr2font_dataset = attr2font_dataset
    self.chars = string.ascii_lowercase + string.ascii_uppercase
    self.dataAdopter: IFontDataAdopter = Attr2FontDataAdopter() if self.attr2font_dataset else LargeScaleTagBasedFontDataAdopter()

    self.font_names = []
    self.font_name_and_char = []

    self.preprocess()


  def preprocess(self):
    self.char_to_num = {self.chars[i]: i for i in range(0, len(self.chars))}

    self.font_names = [line.rstrip() for line in open(self.font_names_path, 'r')]
    if (self.ignore_fonts_file_path is not None):
      self.ignore_fonts = set([line.rstrip() for line in open(self.ignore_fonts_file_path, 'r')])
      before_len = len(self.font_names)
      self.font_names = list(filter(lambda font_name: not font_name in self.ignore_fonts, self.font_names))
      after_len = len(self.font_names)
      if (before_len != after_len):
        print('ignore fonts len =', before_len - after_len)

    for font_name in self.font_names:
      for char in self.chars:
        self.font_name_and_char.append([font_name, char])

    random.shuffle(self.font_name_and_char)
    if (self.max_fonts_num >= 0 and len(self.font_name_and_char) > self.max_fonts_num):
      print('clip dataset_len: ', len(self.font_name_and_char), '>>>', self.max_fonts_num)
      self.font_name_and_char = self.font_name_and_char[:self.max_fonts_num]

    image_size = 224

    self.transform = T.Compose([
      T.Resize((image_size, image_size)),
      T.ToTensor(),
      # T.Normalize(mean=[0.485, 0.456, 0.406],
      #              std=[0.229, 0.224, 0.225])
    ])

  def __getitem__(self, index):
    font_name, char = self.font_name_and_char[index]

    image_path = os.path.join(self.image_dir, self.dataAdopter.alphabet_to_img_file_name(font_name, char))
    image = Image.open(image_path)

    # skip_crop = not self.attr2font_dataset
    skip_crop = False

    if (not skip_crop):
      # crop
      image = image.convert('L')
      inv_image = ImageOps.invert(image)
      image = image.crop(inv_image.getbbox())

      # padding
      long_len = max(image.size[0], image.size[1])
      each_img_transform = T.Compose([
          T.Pad(((long_len - image.size[0]) // 2, (long_len - image.size[1]) // 2), fill=255),
      ])
      image = each_img_transform(image)
    image = image.convert('RGB')

    image = self.transform(image)
    label = self.char_to_num[char]

    if (only_lower):
      label = self.to_lower(label)

    return image, label, font_name, char

  def __len__(self):
    return len(self.font_name_and_char)

  def dump_font_names(self, dataset_font_names_path):
    with open(dataset_font_names_path, 'w') as f:
      f.write('\n'.join(self.font_names))

  def to_lower(self, label):
    if label > len(string.ascii_lowercase) - 1:
      label -= len(string.ascii_lowercase)
    return label

def split_dataset_fonts(opts):
  all_fonts_file_name = opts.font_names_file_name
  font_names_path = os.path.join(opts.dataset_root, all_fonts_file_name)
  ignore_fonts_file_path = os.path.join(opts.dataset_root, opts.ignore_fonts_file_name)

  all_font_names = [line.rstrip() for line in open(font_names_path, 'r')]
  if (ignore_fonts_file_path is not None):
    ignore_fonts = set([line.rstrip() for line in open(ignore_fonts_file_path, 'r')])
    all_font_names = list(filter(lambda font_name: not font_name in ignore_fonts, all_font_names))

  if (opts.shuffle):
    random.shuffle(all_font_names)

  if (opts.max_fonts_num < len(all_font_names)):
    all_font_names = all_font_names[:opts.max_fonts_num]

  all_len = len(all_font_names)

  train_test_val = [0.7, 0.2, 0.1]

  test_head_idx = int(all_len*train_test_val[0])
  val_heed_idx = test_head_idx + int(all_len*train_test_val[1])

  train_fonts = all_font_names[:test_head_idx]
  test_fonts = all_font_names[test_head_idx:val_heed_idx]
  val_fonts = all_font_names[val_heed_idx:]
  print(len(all_font_names))
  print(len(train_fonts))
  print(len(test_fonts))
  print(len(val_fonts))


  with open(os.path.join(opts.dataset_root, 'train_font_names.txt'), 'w') as f:
    f.write('\n'.join(train_fonts))
  with open(os.path.join(opts.dataset_root, 'test_font_names.txt'), 'w') as f:
    f.write('\n'.join(test_fonts))
  with open(os.path.join(opts.dataset_root, 'val_font_names.txt'), 'w') as f:
    f.write('\n'.join(val_fonts))


def get_loader(opts, font_names_file_name, shuffle):
  dataset_root_dir = opts.dataset_root
  image_dir = opts.image_dir_name
  attr2font_dataset = opts.is_attr2font_dataset
  ignore_fonts_file_name = opts.ignore_fonts_file_name
  num_workers = opts.num_workers
  batch_size = opts.batch_size

  dataset = FontOcrDataSet(dataset_root_dir, image_dir, font_names_file_name, attr2font_dataset, ignore_fonts_file_name, opts.max_fonts_num)
  data_loader = data.DataLoader(dataset=dataset, drop_last=True, batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
  return data_loader