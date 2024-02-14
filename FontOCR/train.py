from dataloader import chars
import os
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import torch
import datetime
from dataloader import get_loader
import csv


def train(epoch, model, dataloader, device, optimizer, criterion, train_loss, train_acc, counter):
  model.train()

  train_total_loss = 0
  train_total_acc = 0

  for batch_idx, (data, label, _, _) in enumerate(dataloader):
    data, label = data.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    train_total_loss, counter = cal_loss(counter,data.shape[0], train_total_loss ,loss.detach().item())
    train_total_acc += cal_acc(label, output)

  train_loss.append(train_total_loss)
  train_acc.append(train_total_acc/counter)

  now = datetime.datetime.now()
  print('[{}] Train Epoch: {} Average loss: {:.6f} Average acc: {:.6f}'.format(
      now,
      epoch, train_total_loss, (train_total_acc / counter)*100))

def start_train(opts):
  epoch_num = opts.n_epochs
  print('train epoch:', epoch_num)

  checkpoint_dir = opts.checkpoint_root
  start_time_dir_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  checkpoint_dir = f'{checkpoint_dir}/{start_time_dir_str}'
  print('checkpoint_dir', checkpoint_dir)

  train_data_loader = get_loader(opts, opts.train_font_names_file_name, True)
  val_data_loader = get_loader(opts, opts.val_font_names_file_name, False)

  model = resnet18(weights=ResNet18_Weights.DEFAULT)
  device=torch.device('cuda')

  model.fc = nn.Linear(model.fc.in_features, len(chars))

  model = model.to(device)
  model.train()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

  val_total_loss = 0
  val_total_acc = 0
  counter = 0

  train_loss = []
  train_acc = []
  val_loss = []
  val_acc = []

  print("start train")
  for epoch in range(epoch_num):
    val_total_loss = 0
    val_total_acc = 0
    counter = 0

    train(epoch, model, train_data_loader, device, optimizer, criterion, train_loss, train_acc, counter)
    counter = 0

    if (not os.path.exists(checkpoint_dir)):
      os.makedirs(checkpoint_dir, exist_ok=True)
    val(val_data_loader, model, device, epoch, val_total_loss, val_total_acc, counter, val_loss, val_acc)
    weight_file_path = os.path.join(checkpoint_dir, f"{epoch}.pth")
    torch.save(model.state_dict(), weight_file_path)

  with open(os.path.join(checkpoint_dir, 'loss_and_acc.csv'), 'w') as f:
    writer = csv.writer(f)
    head = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    writer.writerow(head)

    for epoch, (t_loss, t_acc, v_loss, v_acc) in enumerate(zip(train_loss, train_acc, val_loss, val_acc)):
      writer.writerow([epoch, t_loss, t_acc.detach().item(), v_loss, v_acc.detach().item()])

def cal_loss(n,batch,total_loss,loss):
  return (total_loss * n + loss*batch)/(n + batch), n + batch

def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg)

def val(dataloader, model, device, epoch,  val_total_loss, val_total_acc, counter, val_loss, val_acc):
  model.eval()
  criterion = nn.CrossEntropyLoss()

  with torch.no_grad():
    for batch_idx, (data, label, font_name, char) in enumerate(dataloader):
      data = data.to(device)
      label = label.to(device)
      output = model(data)
      loss = criterion(output,label)
      val_total_loss , counter = cal_loss(counter,data.shape[0],val_total_loss ,loss.detach().item())
      val_total_acc += cal_acc(label,output)
      # now = datetime.datetime.now()
      # if batch_idx % 1000 == 0:
      #   print('[{}] {:5} Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}\tAverage acc: {:.6f}'.format(
      #     now,
      #     'Val',
      #     epoch, batch_idx * len(data), len(dataloader.dataset),
      #     100. * batch_idx / len(dataloader), val_total_loss, (val_total_acc / counter)*100))
    now = datetime.datetime.now()

    print('[{}] {:5} Epoch: {} Average loss: {:.6f} Average acc: {:.6f}'.format(
      now,
      'Val',
      epoch, val_total_loss, (val_total_acc / counter)*100))
    val_loss.append(val_total_loss)
    val_acc.append(val_total_acc/counter)