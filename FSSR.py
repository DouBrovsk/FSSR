import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import time
import copy
import torchvision.io
import warnings

#os.chdir(os.path.dirname(os.path.realpath(__file__)))
from models import *
from meta import Meta
import utils
from loss_functions import perceptionLoss, ultimateLoss
from finetuner import FineTuner
warnings.filterwarnings("ignore", message="torch.gels is deprecated in favour of")

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## General purpose functions
def MAMLtrain(logger_name,model, epochs_nb, trainloader, validloader, batch_size=1, verbose=True):
    
    torch.cuda.empty_cache()
    logger = utils.Logger('%s.log' % logger_name)
    print('Logger:[%s]' % logger_name, file=logger)
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 6500000.0
    train_size = len(trainloader)
    valid_size = len(validloader)
    print("Training start", flush=True)

    for epoch in range(epochs_nb):
        # Verbose 1
        if verbose:
            print("Epoch [" + str(epoch + 1) + " / " + str(epochs_nb) + "]")
            print("-" * 10, flush=True)

        # Training
        running_loss = 0.0
        verbose_loss = 0.0
        for i,data in enumerate(trainloader):
            
            
            support_data = data[0].to(device)
            support_label= data[1].to(device)
            query_data = data[2].to(device)
            query_label = data[3].to(device)
            
            
            loss = model(support_data, support_label, query_data, query_label)
            print("Training Loss : " + str(loss), flush=True,file=logger)

            if i % 20 == 0:
                print("Batch " + str(i) + " / " + str(int(train_size)), flush=True)
                
            running_loss += loss
            verbose_loss += loss
            
            if i % 100 == 0 and i != 0:
                print("Loss over last 100 batches: " + str(verbose_loss / (100 * batch_size)), flush=True,file=logger)
                verbose_loss = 0.0

        # Verbose 2
        if verbose:
            epoch_loss = running_loss / (train_size * batch_size)
            print(" ", flush=True)
            print(" ", flush=True)
            print("****************")
            print('Training Loss over the epoch: {:.7f}'.format(epoch_loss), flush=True,file=logger)
        # Validation
        running_loss = 0.0
        verbose_loss = 0.0
        for i, data in enumerate(validloader):
            
            support_data = data[0].to(device)
            support_label= data[1].to(device)
            query_data = data[2].to(device)
            query_label = data[3].to(device)
            
            loss = model.finetuning(support_data, support_label, query_data, query_label)

            running_loss += loss

        # Verbose 3
        if verbose:
            epoch_loss = running_loss / (valid_size * batch_size)
            print('Validation Loss: {:.7f}'.format(epoch_loss), flush=True,file=logger)

        # Copy the model if it gets better with validation
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model.state_dict())
    # Verbose 4
    if verbose:
        time_elapsed = time.time() - since
        print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60), flush=True)
        print("Best validation loss: " + str(best_loss), flush=True)

    model.load_state_dict(best_model)  # In place anyway
    return model  # Returning just in case


def makeCheckpoint(model, save_path, verbose=True):  # Function to save weights
    torch.save(model.state_dict(), save_path)
    if verbose:
        print("Weights saved to: " + save_path, flush=True)
    return


### Different Modes
def meta_train(logger_name,train_path, valid_path, batch_size,num_shot, epoch_nb, learning_rate, meta_learning_rate, save_path, 
               verbose, weights_load=None, loss_func='MSE', loss_network='vgg16', network='EDSR'):

    ## Init training
    scale_factor = 2

    # Setup model and hyper parameters
    if network == 'EDSR':
        autoencoder = EDSR(scale=scale_factor)

    config = autoencoder.getconfig()

    meta_learner = Meta(logger_name,config, learning_rate, meta_learning_rate, 10, 10).to(device)

    transform = torchvision.transforms.Compose([transforms.ToTensor()])

    # Data loading
    trainset = utils.DADataset(train_path, transform=transform, num_shot=num_shot, is_valid_file=utils.is_file_not_corrupted, 
                               scale_factor=scale_factor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4) # Batch must be composed of images of the same size if >1
    print("Found " + str(len(trainloader)*batch_size) + " images in " + train_path, flush=True)
    print(len(trainloader))
    print (batch_size)
        
    #print("spt_data.shape = " + trainloader[0][0].shape)
    #print("qry_data.shape = " + trainloader[0][2].shape)

    validset = utils.DADataset(valid_path, transform=transform,num_shot=num_shot,is_valid_file=utils.is_file_not_corrupted, 
                               scale_factor=scale_factor)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("Found " + str(len(validloader)*batch_size) + " images in " + valid_path, flush=True)
    

    if weights_load is not None: # Load weights for further training if a path was given.
        meta_learner.load_state_dict(torch.load(weights_load))
        print("Loaded weights from: " + str(weights_load), flush=True)

    #print(autoencoder, flush=True)

    del autoencoder

    # Start training
    meta_learner = MAMLtrain(logger_name,meta_learner, epoch_nb, trainloader, validloader, batch_size=batch_size)
    makeCheckpoint(meta_learner, save_path)
    return

## Upscale - Using the model
def MAMLupscale(in_path, out_path, weights_path, learning_rate, batch_size, verbose, device_name, benchmark=False, network='EDSR'):
    if device_name == "cuda_if_available":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    if network == 'EDSR':
        model = EDSR()

    config = model.getconfig()

    meta_learner = Meta(config, learning_rate, 0, 10, 10) # Meta learning rate = 0 so no update is performed at test time. 
    #Anyway, it doesn't matter which value is given here because it won't be used at test time.
    meta_learner.load_state_dict(torch.load(weights_path))

    del model

    scale_factor = 2
    transform = transforms.ToTensor()
    testset = utils.FSDataset(in_path, transform=transform, is_valid_file=utils.is_file_not_corrupted, scale_factor=scale_factor, mode='train')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    n = len(testloader)
    if verbose:
        print("Found " + str(n) + " images in " + in_path, flush=True)
        print("Beginning upscaling...", flush=True)
        print("Clock started ", flush=True)

    since = time.time()
    for i, data in enumerate(testloader):

        support_data, support_label, query_data, query_label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        query_label = query_label.squeeze(0)
        support_data, support_label = support_data.squeeze(0), support_label.squeeze(0)

        if benchmark:
            pass # ToDo: divide by 2 and then re-upscale
        else:
            pass
        reconstructed, reconstructed_without = meta_learner.test(support_data, support_label, query_data)
        img = transforms.Compose([torchvision.transforms.ToPILImage(mode='RGB')])(reconstructed[0].cpu())
        img_l = transforms.ToPILImage(mode='RGB')(query_label.cpu())
        img_without = transforms.ToPILImage(mode='RGB')(reconstructed_without[0].cpu())
        if not(os.path.exists(os.path.join(out_path, 'without/'))):
            os.mkdir(os.path.join(out_path, 'without/'))
        if not(os.path.exists(os.path.join(out_path, 'labels/'))):
            os.mkdir(os.path.join(out_path, 'labels/'))
        img_without.save(os.path.join(os.path.join(out_path, 'without/'), str(i) + '.png'))
        img_l.save(os.path.join(os.path.join(out_path, 'labels/'), str(i) + '.png'))
        img.save(os.path.join(out_path, str(i) + ".png"))
        if verbose:
            if i % 100 == 0:
                print("Image " + str(i) + " / " + str(n), flush=True)

    time_elapsed = time.time() - since
    if verbose:
        print("Processed " + str(n) + " images in " + "{:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60), flush=True)
        print("Overall speed: " + str(n/time_elapsed) + " images/s", flush=True)
        print("Upscaling: Done, files saved to " + out_path, flush=True)

    return time_elapsed, n/time_elapsed, n, scale_factor


def finetuneMaml(train_path, valid_path, batch_size, epoch_nb, learning_rate, meta_learning_rate, load_weights, save_weights, finetune_depth, network='EDSR'):
    if network == 'EDSR':
        model = EDSR().to(device)
    config = model.getconfig()
    # Data loading
    transform = transforms.ToTensor()
    scale_factor = 2
    trainset = utils.DADataset(train_path, transform=transform, num_shot=10, is_valid_file=utils.is_file_not_corrupted,
                               scale_factor=scale_factor, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=4)  # Batch must be composed of images of the same size if >1
    print("Found " + str(len(trainloader) * batch_size) + " images in " + train_path, flush=True)

    validset = utils.DADataset(valid_path, transform=transform,num_shot=10, is_valid_file=utils.is_file_not_corrupted,
                               scale_factor=scale_factor, mode='train')
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Found " + str(len(validloader) * batch_size) + " images in " + valid_path, flush=True)
    
    meta_learner = Meta(config, learning_rate, meta_learning_rate, 10, 10, load_weights=load_weights).to(device)
    meta_learner = MAMLtrain(meta_learner, epoch_nb, trainloader, validloader, batch_size)
    makeCheckpoint(meta_learner, save_weights)

    # ToDo: Rework Data Augmentation.
    # ToDo: Then, maybe modify meta.py code to be able to only train on the latest layers of the neural network.

    return

def model_train(logger_name,train_path, valid_path, epoch_nb=1, batch_size=1, load_weights=None, save_weights='weights.pt', model_name='EDSR'):
    verbose = True
    logger = utils.Logger('%s.log' % logger_name)
    print('Logger:[%s]' % logger_name, file=logger)
    
    
    if model_name == 'EDSR':
        super_res_model = EDSR().to(device)

    if load_weights is not None:
        super_res_model.load_state_dict(torch.load(load_weights))
        print("Loaded weights from: " + str(load_weights), flush=True)

    def train(model, epochs_nb, trainloader, validloader, optimizer):
        since = time.time()
        best_model = copy.deepcopy(model.state_dict())
        best_loss = 6500000.0
        train_size = len(trainloader)
        valid_size = len(validloader)
        print("Training start", flush=True)
        
        
        #first validation pass to get initial validation loss (util when FT)
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validloader):
                query, label =  data[2].to(device), data[3].to(device)
                query = model(query)
                loss = F.mse_loss(query, label)
                running_loss += loss.item()
            # Verbose 3
            if verbose:
                epoch_loss = running_loss / (valid_size*batch_size)
                print('Initial Validation Loss: {:.7f}'.format(epoch_loss), flush=True,file=logger)

        
        

        for epoch in range(epochs_nb):
            # Verbose 1
            if verbose:
                print("Epoch [" + str(epoch+1) + " / " + str(epochs_nb) + "]", flush=True)
                print("-" * 10, flush=True)

            # Training
            running_loss = 0.0
            verbose_loss = 0.0
            for i, data in enumerate(trainloader):
                query, label = data[2].to(device), data[3].to(device)
                optimizer.zero_grad()
                query = model(query)
                loss = F.mse_loss(query, label)
                loss.backward()
                optimizer.step()
                print(loss.item())

                if i%100 == 0:
                    print("Batch " + str(i) + " / " + str(int(train_size)), flush=True)
                running_loss += loss.item()
                verbose_loss += loss.item()
                if i% 100 == 0 and i !=0:
                    print("Loss over last 100 batches: " + str(verbose_loss/(100*batch_size)), flush=True,file=logger)
                    verbose_loss = 0.0
                    

            # Verbose 2
            if verbose:
                epoch_loss = running_loss / (train_size*batch_size)
                print(" ", flush=True)
                print(" ", flush=True)
                print("****************")
                print('Training Loss: {:.7f}'.format(epoch_loss), flush=True,file=logger)
            # Validation
            running_loss = 0.0
            verbose_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(validloader):
                    query, label =  data[2].to(device), data[3].to(device)
                    query = model(query)
                    loss = F.mse_loss(query, label)
                    running_loss += loss.item()

                # Verbose 3
                if verbose:
                    epoch_loss = running_loss / (valid_size*batch_size)
                    print('Validation Loss: {:.7f}'.format(epoch_loss), flush=True,file=logger)

            # Copy the model if it gets better with validation
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
        # Verbose 4
        if verbose:
            time_elapsed = time.time() - since
            print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60), flush=True)
            print("Best validation loss: " + str(best_loss), flush=True)

        model.load_state_dict(best_model) # In place anyway
        return model # Returning just in case

    def makeCheckpoint(model, save_path): # Function to save weights
        torch.save(model.state_dict(), save_path)
        if verbose:
            print("Weights saved to: " + save_path, flush=True)
        return

    trainset = utils.DADataset(train_path, num_shot=1, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validset = utils.DADataset(valid_path,num_shot=1, transform=transforms.ToTensor())
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(super_res_model.parameters(), lr=0.0001, amsgrad=True)
    super_res_model = train(super_res_model, epoch_nb, trainloader, validloader, optimizer)
    makeCheckpoint(super_res_model, save_weights)

    return

def upscale(load_weights, input, out):
    edsr = EDSR().to(device)
    print(edsr)

    if load_weights is not None:
        edsr.load_state_dict(torch.load(load_weights))
        print("Loaded weights from: " + str(load_weights), flush=True)
    label_path = os.path.join(out, 'labels/')
    if not(os.path.exists(label_path)):
        os.mkdir(label_path)

    batch_size = 1
    validset = utils.FSDataset(input, transform=transforms.ToTensor())
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for i, data in enumerate(validloader):
            query, label = data[2].to(device), data[3].to(device)
            query = edsr(query)
            query = transforms.ToPILImage(mode='RGB')(query.squeeze(0).cpu())
            query.save(os.path.join(out, str(i) + '.png'))
            label = transforms.ToPILImage(mode='RGB')(label.squeeze(0).cpu())
            label.save(os.path.join(label_path, str(i) + '.png'))
    return


