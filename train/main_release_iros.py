# Code is adapted from ERFNet model - https://github.com/Eromera/erfnet_pytorch
# Sept 2018
# Sudhir Kumar

#######################

import os
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Pad
from torchvision.transforms import ToTensor, ToPILImage

from piwise.dataset import VOC12,cityscapes
from piwise.criterion import CrossEntropyLoss2dv2
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.visualize import Dashboard
from piwise.ModelDataParallel import ModelDataParallel,CriterionDataParallel #https://github.com/pytorch/pytorch/issues/1893

import importlib
import evalIoU

from shutil import copyfile

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#np.set_printoptions(threshold='nan')

# This Module is used to segment out the road defects
class defectSegmentationNet(nn.Module):
    def __init__(self, num_classes,model):  #use encoder to pass pretrained encoder
        super().__init__()
        
        self.encoder = list(model.children())[0].encoder
        self.decoder = nn.Sequential(*list(list(model.children())[0].decoder.layers.children())) 
        self.decoder_last_child = list(list(model.children())[0].decoder.children())[-1]
        self.output_conv = nn.ConvTranspose2d( 16, 10, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        x2 = self.output_conv(x)
        return x2

# This module combines road segmentation module and road defect segmentation module    
class cascadeNet(nn.Module):
    def __init__(self, base_model, fine_grained_model):  #use encoder to pass pretrained encoder
        super().__init__()
        self.base_model = base_model
        self.fine_grained_model = fine_grained_model
        
    def forward(self, input):
        x = self.base_model(input)
        x = x.max(1)[1]
        x[x != 0] = 255
        x[x == 0] = 1
        x[x == 255] = 0
        x = torch.stack((x,x,x),1).float()
        inputs = (input.squeeze(0)*x.squeeze(0))
        if len(inputs) == 3 :
            inputs = inputs.unsqueeze(0)
        x = self.fine_grained_model(inputs)
        return x
    
NUM_CHANNELS = 3
NUM_CLASSES = 10 

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

target_transform = Compose([
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])

# The dataset has 0-18 labels as mentioned in cityscapes dataset and correspondingly labeled with pothole as 19, waterlog as 20, muddyroad as 21, obstruction as 22, cementroad as 23, rough road patch as 24, wet road patch as 25, side road as 26, bumps as 27.
# Also the folder structure of dataset is same as cityscapes.
# As mentioned in the paper various hierachies in the dataset can be obtained by relabeling and combining multiple labels to a single label.

# So here we relabel all the 0-18 labels of cityscapes as 255 and change road defect labels from 1 - 9 respectively.

#Augmentations - different function implemented to perform random augments on both image and target

class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Scale(self.height, Image.BILINEAR)(input)
        target = Scale(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

        input = ToTensor()(input)
        
        target = ToLabel()(target)
        
        for iter in range(1,19):
            target = Relabel(iter, 255)(target)
        
        target = Relabel(19, 1)(target) 
        target = Relabel(20, 2)(target)
        target = Relabel(21, 3)(target)
        target = Relabel(22, 4)(target)
        target = Relabel(23, 5)(target)
        target = Relabel(24, 6)(target)
        target = Relabel(25, 7)(target)
        target = Relabel(26, 8)(target)
        target = Relabel(27, 9)(target) 
        
        return input, target

best_acc = 0
save_val = False

def train(args, model, enc):
    global best_acc

    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    weight = torch.ones(NUM_CLASSES)
    weight[0] = 1
    weight[1] = 1
    weight[2] = 1
    weight[3] = 1
    weight[4] = 1
    weight[5] = 1
    weight[6] = 1
    weight[7] = 1
    weight[8] = 1
    weight[9] = 1
        
    
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    #Loading the dataset
    co_transform = MyCoTransform(False, augment=True, height=args.height)#1024)
    co_transform_val = MyCoTransform(False, augment=False, height=args.height)#1024)
    
    
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        criterion = CrossEntropyLoss2dv2(weight.cuda())
        
    else:
        criterion = CrossEntropyLoss2dv2(weight)

    savedir = '../save/'+args.savedir

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))
    
    # We use Adam optimizer with lr of 5e-4
    optimizer = Adam([ {'params' : model.parameters()},], 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    
    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'#'/model_best.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
    
    
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)                            ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    cont_train_loss = []
    cont_val_loss = []
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        #TODO: remake the evalIoU.py code to avoid using "evalIoU.args"
        confMatrix    = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        
        model.train()
    
        for step, (images,oldimages, labels, filename, filenameGt) in enumerate(loader):
            start_time = time.time()
            break
            
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                #compatibility with criterion dataparallel
                if isinstance(outputs, list):   #merge gpu tensors
                    outputs_cpu = outputs[0].cpu()
                    for i in range(1,len(outputs)):
                        outputs_cpu = torch.cat((outputs_cpu, outputs[i].cpu()), 0)
                else:
                    outputs_cpu = outputs.cpu()

                #start_time_iou = time.time()
                for i in range(0, outputs_cpu.size(0)):   #args.batch_size
                    prediction = ToPILImage()(outputs_cpu[i].max(0)[1].data.unsqueeze(0).byte())
                    groundtruth = ToPILImage()(labels[i].cpu().byte())
                    nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth, confMatrix, perImageStats, evalIoU.args)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)


        if not args.eval:    
            average_epoch_loss_train = 0#sum(epoch_loss) / len(epoch_loss)
        else :
            average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        #evalIoU.printConfMatrix(confMatrix, evalIoU.args)
        
        iouTrain = 0
        if (doIouTrain ):
            # Calculate IOU scores on class level from matrix
            classScoreList = {}
            for label in evalIoU.args.evalLabels:
                labelName = evalIoU.trainId2label[label].name
                classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)
            print(classScoreList)
            iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol

            iouTrain = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
            print ("EPOCH IoU on TRAIN set: ", iouAvgStr)
            
            evalIoU.printClassScoresPytorchTrain(classScoreList, evalIoU.args)
            print("--------------------------------")
            print("Score Average : " + iouAvgStr )#+ "    " + niouAvgStr)
            print("--------------------------------")
            

        #Validate on val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        #model = pretrained_model
        epoch_loss_val = []
        time_val = []

        #New confusion matrix data
        confMatrix    = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0
        val_ct = 0
        for step, (images, oldimages, labels, filename, filenameGt) in enumerate(loader_val):
            start_time = time.time()
            #break
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)
            outputs = model(inputs) 
            
            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)
            
            #Add outputs to confusion matrix
            if (doIouVal):
                #compatibility with criterion dataparallel
                if isinstance(outputs, list):   #merge gpu tensors
                    outputs_cpu = outputs[0].cpu()
                    for i in range(1,len(outputs)):
                        outputs_cpu = torch.cat((outputs_cpu, outputs[i].cpu()), 0)
                else:
                    outputs_cpu = outputs.cpu()
                    targets_cpu = targets.cpu()
                    
                start_time_iou = time.time()
                for i in range(0, outputs_cpu.size(0)):   #args.batch_size
                    val_ct += 1
                    pred_img = outputs_cpu[i].max(0)[1].data.unsqueeze(0)
                    
                    prediction = ToPILImage()(pred_img.byte())
                    
                    
                    groundtruth = ToPILImage()(labels[i].cpu().byte())
                    
                    nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth, confMatrix, perImageStats, evalIoU.args)
                print ("Time to add confusion matrix: ", time.time() - start_time_iou)
                       
        
        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        
        print(doIouVal)
        
        # Calculate IOU scores on class level from matrix
        iouVal = 0
        confMatrix= confMatrix[:12,:12]
        
        if (doIouVal):
            #start_time_iou = time.time()
            classScoreList = {}
            for label in evalIoU.args.evalLabels:
                labelName = evalIoU.trainId2label[label].name
                classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)
            print(classScoreList)

            iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol
            iouVal = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
            print ("EPOCH IoU on VAL set: ", iouAvgStr)
            #print("")
            #evalIoU.printClassScoresPytorchTrain(classScoreList, evalIoU.args)
            #print("--------------------------------")
            #print("Score Average : " + iouAvgStr )#+ "    " + niouAvgStr)
            #print("--------------------------------")
            #print("")
            #print ("Time to calculate confusion matrix: ", time.time() - start_time_iou)
            #input ("Press key to continue...")
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
    
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        
        filename = savedir+'/model-'+str(epoch)+'}.pth'
        filenamebest = savedir+'/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print('save: {'+filename+'} (epoch: {'+str(epoch)+'})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print('save: {'+filenamebest+'} (epoch: {'+str(epoch)+'})')
            
            
            with open(savedir + "/best_encoder.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def evaluate(args, model):
    model.eval()

    image = input_transform(Image.open(args.image))
    label = model(Variable(image, volatile=True).unsqueeze(0))
    #label = color_transform(label[0].data.max(0)[1])

    image_transform(label).save(args.label)

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model
        
def main(args):
    savedir = '../save/'+args.savedir

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(20)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        #model = ModelDataParallel(model).cuda()
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
 
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    #train(args, model)
    print("========== TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
        else:
            pretrainedEnc = next(model.children()).encoder
        model = model_file.Net(20, encoder=pretrainedEnc)  #Add decoder to encoder
        model = model.cuda()
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model
        
        defSegNet = defectSegmentationNet(20,model)
    
        #defSegNet = torch.nn.DataParallel(defSegNet).cuda()
    
        road_segNet_file = importlib.import_module('erfnet2')
        roadSeg_model = road_segNet_file.Net(20)
        #roadSeg_model = torch.nn.DataParallel(roadSeg_model).cuda()

        cascadeNet_model = cascadeNet(roadSeg_model,defSegNet)
        cascadeNet_model = torch.nn.DataParallel(cascadeNet_model).cuda()
        cascadeNet_model = load_my_state_dict(cascadeNet_model,torch.load('../save/release_version_test/model_best.pth')) 

        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, cascadeNet_model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="erfnet_multiscale")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--eval', action='store_true',default=False)
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes a lot to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=False) #calculating IoU takes about 0,10 seconds per image ~ 50s per 500 images in VAL set, so 50 extra secs per epoch    
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
