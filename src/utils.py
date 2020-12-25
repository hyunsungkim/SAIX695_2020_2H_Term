import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

def get_prototype(ebd, label):
    pass

def distance_metric(x, targets):
    distances = torch.zeros(len(targets))
    for i, target in enumerate(targets):
        #distance = square_euclidean_metric(x, target)
        distances[i] = torch.norm(x-target, 'fro')
    return distances


def loss_fn(queries, proto_shots, label):
    losses = torch.zeros(queries.size(0))
    predictions = []
    for i, query in enumerate(queries):
        # For each query, get distance to all classes
        distances = distance_metric(query, proto_shots)
        
        #print("DEBUG distance.shape\t", distances.shape)
        # Get softmax loss for each query
        loss = torch.nn.functional.softmax(distances, dim=0)
        #print("DEBUG loss.shape\t", loss.shape)
        losses[i] = loss[label]

        # Get prediction from softmax value
        pred = torch.argmax(loss)
        predictions.append(pred)

    # Aggregate losses of all queries
    losses = -1*torch.log(torch.mean(losses))
    #losses = torch.mean(losses)
    return losses, predictions
    

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
    #pred = torch.argmax(logits, dim=1)
    pred = logits.detach().cpu()
    label_ = label.detach().cpu()
    return (pred == label_).type(torch.cuda.FloatTensor).mean().item()


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
