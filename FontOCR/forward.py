import os
from torchvision.models import resnet18
import torch.nn as nn
import torch
from dataloader import FontOcrDataSet, chars
from collections import defaultdict
from torch.utils import data
import pandas as pd
from tqdm import tqdm

def start_forward(opts):
  dataset = FontOcrDataSet(opts.dataset_root, opts.image_dir_name, opts.forward_font_names_file_name, opts.is_attr2font_dataset, opts.ignore_fonts_file_name, opts.max_fonts_num)

  print('dataset_num=', len(dataset))
  data_loader = data.DataLoader(dataset=dataset, drop_last=False, batch_size=1,
                                  shuffle=False,
                                  num_workers=8)
  model = resnet18()
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model.fc = nn.Linear(model.fc.in_features, len(chars))

  model.eval()
  model.load_state_dict(torch.load(opts.load_weight_path))
  model = model.to(device)

  loss_dict = defaultdict(lambda: defaultdict(int))

  model.eval()
  criterion = nn.CrossEntropyLoss()

  softmax = nn.Softmax(dim=1)

  with torch.no_grad():
    for batch_idx, (image, target, font_name, char) in enumerate(tqdm(data_loader)):
      forward(image, target, font_name, char, model, criterion, loss_dict, device, softmax)

  # print_dict(loss_dict, dataset.font_names)
  loss_dict2csv_softmax(opts, loss_dict, dataset.font_names)

def forward(image, target, font_name, char, model, criterion, loss_dict, device, softmax):
  # print(type(image), device)
  image = image.to(device)
  target = target.to(device)
  output = model(image)
  # loss = criterion(output, target)
  # loss_dict[font_name[0]][char[0]] = float(loss)
  loss_dict[font_name[0]][char[0]] = softmax(output).tolist()

def print_dict(loss_dict: defaultdict, font_names):
  for font_name in font_names:
    print(font_name)
    for char in chars:
      print(loss_dict[font_name][char])
    print()

def loss_dict2csv(opts, loss_dict, font_names):
  print(loss_dict)
  file_path = os.path.join(opts.output_loss_csv_dir, opts.output_loss_csv_path)
  with open(file_path, 'w') as f:
    head = ['font_name', *list(chars)]
    f.write(','.join(head)+'\n')

    for font_name in font_names:
      row = [font_name]
      for char in chars:
        row.append(str(loss_dict[font_name][char]))
      f.write(','.join(row)+'\n')

def loss_dict2csv_softmax(opts, loss_dict, font_names):
  dir_path = os.path.join(opts.output_loss_csv_dir, 'softmax')
  if (not os.path.exists(dir_path)):
    os.mkdir(dir_path)
  head = ['Actual\\Pred',*list(chars)]
  for font_name in font_names:
    file_path = os.path.join(dir_path, f'{font_name}.csv')
    with open(file_path, 'w') as f:
      f.write(','.join(head)+'\n')

      for char in chars:
        values = loss_dict[font_name][char][0]
        values = list(map(str,values))
        f.write(','.join([char, *values])+'\n')
