from dataloader import chars
import os
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import torch
import datetime
from dataloader import get_loader
import csv

def cal_loss(n,batch,total_loss,loss):
  return (total_loss * n + loss*batch)/(n + batch), n + batch

def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg)

def start_test(opts):
    print('start test')
    pred_list = []
    true_list = []
    start_time_dir_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    test_total_acc = 0
    test_data_loader = get_loader(opts, opts.test_font_names_file_name, False)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    device=torch.device('cuda')

    model.fc = nn.Linear(model.fc.in_features, len(chars))

    model = model.to(device)
    model.load_state_dict(torch.load(opts.load_weight_path))
    model.eval()

    with torch.no_grad():
      for n, (data,label, _, _) in enumerate(test_data_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        test_total_acc += cal_acc(label,output)
        pred = torch.argmax(output , dim =1)
        pred_list += pred.detach().cpu().numpy().tolist()
        true_list += label.detach().cpu().numpy().tolist()
    print(f"test acc:{test_total_acc/len(test_data_loader.dataset)*100}")


    output_dir = os.path.join('test_output', start_time_dir_str)
    if(not os.path.exists(output_dir)):
       os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'pred_and_true.csv'), 'w') as f:
      writer = csv.writer(f)
      head = ['pred', 'true']
      writer.writerow(head)

      for pred_data, true_data in zip(pred_list, true_list):
        writer.writerow([pred_data, true_data])

    with open(os.path.join(output_dir, "opts.txt"), "w") as f:
      for key, value in vars(opts).items():
          f.write(str(key) + ": " + str(value) + "\n")
