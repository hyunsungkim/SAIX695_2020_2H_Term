import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

def step(model, data_shot, data_query, labels, args):
    k = args.nway * args.kshot

    labels_list = labels.tolist()
    labels_num = len(set(labels_list))

    # Embedding
    data = torch.cat([data_shot, data_query], dim=0)
    output = model(data)
    ebd_shot, ebd_query = output[:k], output[k:]
    print(output.shape)

    # Prototype
    proto_shots = torch.zeros([labels_num, ebd_shot.size(1)]).cuda()
    for i in range(labels_num):  # Get prototypes of each class from shot
        shots = ebd_shot[i*args.kshot:(i+1)*args.kshot]
        proto_shots[i] = torch.mean(shots, dim=0)

    # Distance
    #ebd_query = ebd_query.squeeze().unsqueeze(1)
    #proto_shots = proto_shots.squeeze().expand(ebd_query.size(0),-1,-1)
    #distance = torch.sum(torch.square(ebd_query-proto_shots), dim=-1).squeeze()
    #distance = torch.norm(distance, 'fro', dim=-1).squeeze()
    print(ebd_query.shape, proto_shots.shape)
    distance = square_euclidean_metric(ebd_query, proto_shots)

    logits = distance.tolist()
    distance = -F.log_softmax(-distance, dim=-1)

    # Loss and prediction
    predictions = torch.argmax(distance, dim=1)

    loss = distance[:,labels]
    loss = torch.mean(loss)

    logits = distance

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
    #pred = torch.argmin(logits, dim=1)

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
