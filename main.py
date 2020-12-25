import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import CUB as Dataset
from src.sampler import Sampler
from src.train_sampler import Train_Sampler
from src.utils import count_acc, Averager, csv_write, square_euclidean_metric
from src.utils import loss_fn
from model import FewShotModel 

from src.test_dataset import CUB as Test_Dataset
from src.test_sampler import Test_Sampler
" User input value "
TOTAL = 10000  # total step of training
PRINT_FREQ = 1  # frequency of print loss and accuracy at training step
VAL_FREQ = 100  # frequency of model eval on validation dataset
SAVE_FREQ = 100  # frequency of saving model
TEST_SIZE = 200  # fixed

" fixed value "
VAL_TOTAL = 100

def Test_phase(model, args, k):
    model.eval()

    csv = csv_write(args)

    dataset = Test_Dataset(args.dpath)
    test_sampler = Test_Sampler(dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    test_loader = DataLoader(dataset=dataset, batch_sampler=test_sampler, num_workers=8, pin_memory=True)

    print('Test start!')
    for i in range(TEST_SIZE):
        for episode in test_loader:
            data = episode.cuda()

            data_shot, data_query = data[:k], data[k:]

            """ TEST Method """
            """ Predict the query images belong to which classes
            
            At the training phase, you measured logits. 
            The logits can be distance or similarity between query images and 5 images of each classes.
            From logits, you can pick a one class that have most low distance or high similarity.
            
            ex) # when logits is distance
                pred = torch.argmin(logits, dim=1)
            
                # when logits is prob
                pred = torch.argmax(logits, dim=1)
                
            pred is torch.tensor with size [20] and the each component value is zero to four
            """

            # save your prediction as StudentID_Name.csv file
            csv.add(pred)

    csv.close()
    print('Test finished, check the csv file!')
    exit()


def train(args):
    # the number of N way, K shot images
    k = args.nway * args.kshot

    # Train data loading
    dataset = Dataset(args.dpath, state='train')
    train_sampler = Train_Sampler(dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    data_loader = DataLoader(dataset=dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    # Validation data loading
    val_dataset = Dataset(args.dpath, state='val')
    val_sampler = Sampler(val_dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    val_data_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    """ TODO 1.a """
    " Make your own model for Few-shot Classification in 'model.py' file."

    # model setting
    model = FewShotModel()
    """ TODO 1.a END """

    # pretrained model load
    if args.restore_ckpt is not None:
        state_dict = torch.load(args.restore_ckpt)
        model.load_state_dict(state_dict)

    model.cuda()
    model = torch.nn.DataParallel(model, device_ids = range(0,8))
    model.train()

    if args.test_mode == 1:
        Test_phase(model, args, k)

    """ TODO 1.b (optional) """
    " Set an optimizer or scheduler for Few-shot classification (optional) "

    # Default optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    """ TODO 1.b (optional) END """

    tl = Averager()  # save average loss
    ta = Averager()  # save average accuracy

    # training start
    print('train start')
    for epoch in range(TOTAL):
        for episode in data_loader:
            optimizer.zero_grad()

            data, label = [_.cuda() for _ in episode]  # load an episode

            # split an episode images and labels into shots and query set
            # note! data_shot shape is ( nway * kshot, 3, h, w ) not ( kshot * nway, 3, h, w )
            # Take care when reshape the data shot
            data_shot, data_query = data[:k], data[k:]

            label_shot, label_query = label[:k], label[k:]
            label_shot = sorted(list(set(label_shot.tolist())))


            # convert labels into 0-4 values
            label_query = label_query.tolist()
            labels = []
            for j in range(len(label_query)):
                label_idx = label_shot.index(label_query[j])
                labels.append(label_idx)
            labels = torch.tensor(labels).cuda()

            """ TODO 2 ( Same as above TODO 2 ) """
            """ Train the model 
            Input:
                data_shot : torch.tensor, shot images, [args.nway * args.kshot, 3, h, w]
                            be careful when using torch.reshape or .view functions
                data_query : torch.tensor, query images, [args.query, 3, h, w]
                labels : torch.tensor, labels of query images, [args.query]
            output:
                loss : torch scalar tensor which used for updating your model
                logits : A value to measure accuracy and loss
            """

            labels_list = labels.tolist()

            # Embedding
            ebd_shot = model(data_shot)
            ebd_query = model(data_query)

            # Prototype
            proto_shots = torch.zeros([len(set(labels_list)), ebd_shot.size(1), 1]).cuda()

            for i in range(len(set(labels_list))):  # Get prototypes of each class from shot
                shots = ebd_shot[i*args.kshot:(i+1)*args.kshot]
                proto_shots[i] = torch.mean(shots, dim=0)

            # Distance metric
            class_losses = torch.zeros(len(set(labels_list))) # loss for each class
            predictions = torch.zeros(len(ebd_query)) # loss for each queries
            for i in range(len(set(labels_list))):  # Get loss for each class
                # select embedding vectors of i'th class queries
                idx_start = labels_list.index(i)
                cnt = labels_list.count(i)
                query = ebd_query[idx_start:idx_start+cnt]

                class_loss, preds = loss_fn(query, proto_shots, i)

                class_losses[i] = class_loss
                predictions[idx_start:idx_start+cnt] = preds

            #print(class_losses.tolist())
            loss = torch.mean(class_losses)
            #loss = torch.nn.CrossEntropyLoss()(predictions, labels)
            logits = predictions

            """ TODO 2 END """

            acc = count_acc(logits, labels)

            tl.add(loss.item())
            ta.add(acc)

            loss.backward()
            optimizer.step()

            proto = None; logits = None; loss = None

        if (epoch+1) % PRINT_FREQ == 0:
            print('train {}, loss={:.4f} acc={:.4f}'.format(epoch+1, tl.item(), ta.item()))

            # initialize loss and accuracy mean
            tl = None
            ta = None
            tl = Averager()
            ta = Averager()

        # validation start
        if (epoch+1) % VAL_FREQ == 0:
            print('validation start')
            model.eval()
            with torch.no_grad():
                vl = Averager()  # save average loss
                va = Averager()  # save average accuracy
                for j in range(VAL_TOTAL):
                    for episode in val_data_loader:
                        data, label = [_.cuda() for _ in episode]

                        data_shot, data_query = data[:k], data[k:] # load an episode

                        label_shot, label_query = label[:k], label[k:]
                        label_shot = sorted(list(set(label_shot.tolist())))

                        label_query = label_query.tolist()

                        labels = []
                        for j in range(len(label_query)):
                            label = label_shot.index(label_query[j])
                            labels.append(label)
                        labels = torch.tensor(labels).cuda()

                        """ TODO 2 ( Same as above TODO 2 ) """
                        """ Train the model 
                        Input:
                            data_shot : torch.tensor, shot images, [args.nway * args.kshot, 3, h, w]
                                        be careful when using torch.reshape or .view functions
                            data_query : torch.tensor, query images, [args.query, 3, h, w]
                            labels : torch.tensor, labels of query images, [args.query]
                        output:
                            loss : torch scalar tensor which used for updating your model
                            logits : A value to measure accuracy and loss
                        """
                        
                        labels_list = labels.tolist()

                        # Embedding
                        ebd_shot = model(data_shot)
                        ebd_query = model(data_query)

                        # Prototype
                        proto_shots = torch.zeros([len(set(labels_list)), ebd_shot.size(1), 1]).cuda()

                        for i in range(len(set(labels_list))):  # Get prototypes of each class from shot
                            shots = ebd_shot[i*args.kshot:(i+1)*args.kshot]
                            proto_shots[i] = torch.mean(shots, dim=0)

                        # Distance metric
                        class_losses = torch.zeros(len(set(labels_list))) # loss for each class
                        predictions = [] # loss for each queries
                        for i in range(len(set(labels_list))):  # Get loss for each class
                            # select embedding vectors of i'th class queries
                            idx_start = labels_list.index(i)
                            cnt = labels_list.count(i)
                            query = ebd_query[idx_start:idx_start+cnt]
                            # print(f"DEBUG query.shape\t{query.shape}\t{labels_list}")
                            # get loss for this class
                            class_loss, preds = loss_fn(query, proto_shots, i)
            #                print(f"DEBUG class_loss\t{class_loss.shape}\t{class_losses.shape}")
                            class_losses[i] = class_loss
                            predictions.extend(preds)

                        loss = torch.mean(class_losses)
                        logits = torch.tensor(predictions)

                        """ TODO 2 END """

                        acc = count_acc(logits, labels)

                        vl.add(loss.item())
                        va.add(acc)

                        proto = None; logits = None; loss = None

                print('val accuracy mean : %.4f' % va.item())
                print('val loss mean : %.4f' % vl.item())

                # initialize loss and accuracy mean
                vl = None
                va = None
                vl = Averager()
                va = Averager()
            model.train()

        if (epoch+1) % SAVE_FREQ == 0:
            PATH = 'checkpoints/%d_%s.pth' % (epoch + 1, args.name)
            torch.save(model.state_dict(), PATH)
            print('model saved, iteration : %d' % epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model', help="name your experiment")
    parser.add_argument('--dpath', '--d', default='../../term_project/src/dataset/CUB_200_2011/CUB_200_2011', type=str,
                        help='the path where dataset is located')
    parser.add_argument('--restore_ckpt', type=str, help="restore checkpoint")
    parser.add_argument('--nway', '--n', default=5, type=int, help='number of class in the support set (5 or 20)')
    parser.add_argument('--kshot', '--k', default=5, type=int,
                        help='number of data in each class in the support set (1 or 5)')
    parser.add_argument('--query', '--q', default=20, type=int, help='number of query data')
    parser.add_argument('--ntest', default=100, type=int, help='number of tests')
    parser.add_argument('--gpus', type=int, nargs='+', default=1)
    parser.add_argument('--test_mode', type=int, default=0, help="if you want to test the model, change the value to 1")

    args = parser.parse_args()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    #torch.cuda.set_device(args.gpus)

    train(args)

