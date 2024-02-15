import string
import random
from fontGenPage.setup_generator import setup_generator
import torch
import torchvision
from tqdm import tqdm
import csv

from torchvision.utils import save_image

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FontOCR'))
from test_legibility_focr import setup, preprocess, forward, to_lower, chars as all_chars

import numpy as np

def torch_fix_seed(seed=314):
    print('fix seed')
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

torch_fix_seed()

chars = string.ascii_lowercase + string.ascii_uppercase
attr_names = [
"angular",
"artistic",
"attention-grabbing",
"attractive",
"bad",
"boring",
"calm",
"capitals",
"charming",
"clumsy",
"complex",
"cursive",
"delicate",
"disorderly",
"display",
"dramatic",
"formal",
"fresh",
"friendly",
"gentle",
"graceful",
"happy",
"italic",
"legible",
"modern",
"monospace",
"playful",
"pretentious",
"serif",
"sharp",
"sloppy",
"soft",
"strong",
"technical",
"thin",
"warm",
"wide",
]


def start_test():
  (opts, dataloader, generator, attribute_embed, attr_unsuper_tolearn) = setup_generator('.', '.', 'fix')
  # legible_font_index = 491font = 25532index
  legible_font_batch = next(iter(dataloader))

  if opts.test_output_name == 'None':
    output_name = opts.experiment_name
  else:
    output_name = opts.test_output_name
  output_dir = os.path.join('test_legibility_output', output_name)
  print('output_dir:',output_dir)
  if (not os.path.exists(output_dir)):
    os.mkdir(output_dir)

  if (opts.save_test_img):
        img_dir = os.path.join(output_dir, 'imgs')
        if not(os.path.exists(img_dir)):
          os.makedirs(img_dir)
        save_file = os.path.join(img_dir, f"base_font.png")
        save_image(legible_font_batch['img_A'], save_file, nrow=26, normalize=True)

  focr_model, device, softmax = setup(opts)
  labels = list(range(len(string.ascii_lowercase+string.ascii_uppercase)))
  attr_legibility_idx = 23

  attribute_values = []
  for _ in range(opts.test_num):
    attribute_values.append([random.randint(0, 100) for _ in range(opts.attr_channel)])

  for legibility in tqdm([0, 100]):
    outputs = [['fontId', 'actualChar', *all_chars]]
    attr_outputs = [['fontId', *attr_names]]
    pbar = tqdm(total=opts.test_num, leave=False)
    for fontId in range(opts.test_num):
      # batch = dataloader[random.randint(0, len(dataloader)-1)]
      batch = legible_font_batch
      attrB = attribute_values[fontId]
      attrB[attr_legibility_idx] = legibility
      imgs = gen_imgs(opts, attrB, batch, generator, attribute_embed, attr_unsuper_tolearn)

      if (opts.save_test_img):
        img_dir = os.path.join(output_dir, 'imgs', f"legibility_{legibility}")
        if not(os.path.exists(img_dir)):
          os.makedirs(img_dir)
        save_file = os.path.join(img_dir, f"{fontId}.png")
        save_image(imgs, save_file, nrow=26, normalize=True)

      imgs = [torchvision.transforms.ToPILImage()(img) for img in imgs]

      p_imgs = []
      p_labels = []
      for (img, label) in zip(imgs, labels):
        p_img, p_label = preprocess(img, label)
        p_imgs.append(p_img)
        p_labels.append(p_label)

      softmax_output, target = forward(torch.stack(p_imgs), p_labels, focr_model, device, softmax)

      sum = 0
      for target, sm in enumerate(softmax_output):
        sum += sm[to_lower(target)]
        outputs.append([fontId, chars[target], *sm])
      # ave_softmax_output = sum/len(softmax_output)
      # outputs.append([ave_softmax_output, *attrB])
      attr_outputs.append([fontId, *attrB])
      pbar.update(1)
    with open(os.path.join(output_dir, f'softmax_legibility_{legibility}.csv'), 'w') as f:
      writer = csv.writer(f)
      writer.writerows(outputs)
    with open(os.path.join(output_dir, f'attr_legibility_{legibility}.csv'), 'w') as f:
      writer = csv.writer(f)
      writer.writerows(attr_outputs)

def gen_imgs(opts, attrB, batch, generator, attribute_embed, attr_unsuper_tolearn):
  attrB = [val / 100.0 for val in attrB]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  attrB_file = ''
  results_dir = 'test_legibility_outputs'


  with torch.no_grad():
    attrid = torch.tensor([i for i in range(opts.attr_channel)]).to(device)
    attrid = attrid.repeat(52, 1)

    img_A = batch['img_A'].to(device)
    fontembed_A = batch['fontembed_A'].to(device)
    styles_A = batch['styles_A'].to(device)

    attr_A_intensity = attr_unsuper_tolearn(fontembed_A)
    attr_A_intensity = attr_A_intensity.view(attr_A_intensity.size(0), attr_A_intensity.size(2))  # noqa
    attr_A_intensity = torch.sigmoid(3*attr_A_intensity)  # convert to [0, 1]

    attr_B = attrB
    # print('attrB: from dataloader', batch['attr_B'].shape)
    # print('attrB: from file', torch.FloatTensor([attr_B]).to(device).shape)
    attr_B = torch.FloatTensor([attr_B]).to(device)
    # attr_B = batch['attr_B'].to(device) # TODO: 削除
    attr_B_intensity = attr_B

    attr_raw_A = attribute_embed(attrid)
    attr_raw_B = attribute_embed(attrid)

    intensity_A_u = attr_A_intensity.unsqueeze(-1)
    intensity_B_u = attr_B_intensity.unsqueeze(-1)

    attr_A = intensity_A_u * attr_raw_A
    attr_B = intensity_B_u * attr_raw_B

    intensity = attr_B_intensity - attr_A_intensity
    attr = attr_B - attr_A

    # print('img_A', img_A.shape)
    # print('styles_A', styles_A.shape)
    # print('intensity', intensity.shape)
    # print('attr', attr.shape)

    fake_B, _ = generator(img_A, styles_A, intensity, attr)

  return fake_B.data

if __name__ == '__main__':
  start_test()