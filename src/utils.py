import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import model as models

class Scheduler:
    def __init__(self, optimizer, name=None):
        self.name = name
        self.name = 'ExponentialLR'
        self.optimizer = optimizer
        if(self.name == 'ExponentialLR'):
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        self.milestones = range(100,10000,100)
    def step(self, epoch):
        if(self.name == None):
            return None
        elif(epoch in self.milestones):
            self.scheduler.step()
            for param_group in self.optimizer.state_dict()['param_groups']:
                current_lr = param_group['lr']
                print(f"learning_rate {current_lr}")
                return current_lr
        return None

def set_model():
    model = models.RNModel()
    return model


def step(model, data_shot, data_query, labels, args):
    k = args.nway * args.kshot

    labels_list = labels.tolist()
    labels_num = len(set(labels_list))
  #  print(labels_list)

    # Embedding
    data = torch.cat([data_shot, data_query], dim=0)
    output = model(data, args, phase='encode')
    ebd_shot, ebd_query = output[:k], output[k:]

    # Prototype
    proto_shots = torch.zeros([args.nway, ebd_shot.size(1), ebd_shot.size(2), ebd_shot.size(3)]).cuda()
    for i in range(labels_num):  # Get prototypes of each class from shot
        shots = ebd_shot[i*args.kshot:(i+1)*args.kshot]
        proto_shots[i] = torch.mean(shots, dim=0)

    input = torch.zeros([args.query*args.nway, proto_shots.shape[1]+ebd_query.shape[1], ebd_query.shape[2], ebd_query.shape[3]])
   # print(f"input.shape {input.shape}")
    for i in range(args.query):
        for j in range(labels_num):
            input[i*labels_num+j] = torch.cat([proto_shots[j], ebd_shot[i]], dim=0)

    similarity = model(input, args, phase='decode').view(-1,args.nway)
   # print(nn.Softmax(dim=1)(similarity))
    #similarity = nn.Softmax(dim=1)(similarity)
  #  print(f"similarity.shape {similarity.shape}, {similarity}")

    pred = torch.argmax(similarity, dim=1)
    #print(pred)
    loss = torch.nn.CrossEntropyLoss()(similarity, labels)
    #loss = similarity[torch.arange(similarity.shape[0]), labels]
    #print(loss)
    #loss = torch.mean(loss)
    #print("\n\n")
    
    # print(f"Distance\n{distance}")    
    # print(f"Labels\n{labels}")
    # print(loss)
    # print(f"Prediction\n{predictions}\n\n")
    # print("\n\n")
    logits = similarity

    return loss, logits


def square_euclidean_metric(a, b):
    """ Measure the euclidean distance (optional)
    Args:
        a : torch.tensor, features of data query
        b : torch.tensor, mean features of data shots or embedding features

    Returns:
        A torch.tensor, the minus euclidean distance
        between a and b
    """

    n = a.shape[0]
    m = b.shape[0]

    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = torch.pow(a - b, 2).sum(2)

    return logits


def count_acc(logits, label):
    """ In each query set, the index with the highest probability or lowest distance is determined
    Args:
        logits : torch.tensor, distance or probabilty
        label : ground truth

    Returns:
        float, mean of accuracy
    """

    # when logits is distance
    # pred = torch.argmin(logits, dim=1)

    # when logits is prob
    pred = torch.argmax(logits, dim=1)

    # print(pred)
    # print(label)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Averager():
    """ During training, update the average of any values.
    Returns:
        float, the average value
    """

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class csv_write():

    def __init__(self, args):
        self.f = open('20202457_HyunsungKim.csv', 'w', newline='')
        self.write_number = 1
        self.wr = csv.writer(self.f)
        self.wr.writerow(['id', 'prediction'])
        self.query_num = args.query

    def add(self, prediction):

        for i in range(self.query_num):
          self.wr.writerow([self.write_number, int(prediction[i].item())])
          self.write_number += 1

    def close(self):
        self.f.close()
