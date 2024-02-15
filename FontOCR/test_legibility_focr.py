from torchvision.models import resnet18
import torch.nn as nn
import torch
import string
import torchvision.transforms as T
from PIL import ImageOps

only_lower = True
if (only_lower):
  chars = string.ascii_lowercase
else:
  chars = string.ascii_lowercase + string.ascii_uppercase


def setup(opts):
  model = resnet18()
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model.fc = nn.Linear(model.fc.in_features, len(chars))

  model.eval()
  model = model.to(device)
  model.load_state_dict(torch.load(opts.load_weight_path, device))

  # model.eval()

  softmax = nn.Softmax(dim=1)

  return model, device, softmax


def forward(image, target, model, device, softmax):
  # print(type(image), device)
  image = image.to(device)
  output = model(image)
  # loss = criterion(output, target)
  # loss_dict[font_name[0]][char[0]] = float(loss)
  return softmax(output).tolist(), target

def preprocess(image, label):

  only_lower = True
  skip_crop = False

  image_size = 224
  transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

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

  image = transform(image)

  if (only_lower):
    label = to_lower(label)
  return image, label

def to_lower(label):
  if label > len(string.ascii_lowercase) - 1:
    label -= len(string.ascii_lowercase)
  return label