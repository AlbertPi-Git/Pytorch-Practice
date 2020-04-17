import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
from torchvision import datasets, transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# Define device as GPU if available
useCuda=torch.cuda.is_available()
device=torch.device('cuda:0' if useCuda else 'cpu')
torch.backends.cudnn.benchmark=True     # If input images have the same size, enable benchmark can improve runtime 

## all params

trainEpochs=5
learningRate=1e-3

# trainset loader params
trainLoadParams = { 'batch_size': 32,
                    'shuffle': True,
                    'num_workers':0}    


#Don't know why set multiple workers to load data in parallel in .py file will give runtime error
#It can be done in .ipynb file, quite weired

# testset loader params                 
testLoadParams = { 'batch_size': 64,
                    'shuffle': True,
                    'num_workers':0}

# Define transformation to transform PIL image to tensor
transforms=transforms.Compose([transforms.ToTensor()])

# Read trainset and load with dataloader
trainSet=datasets.MNIST(root='./MNIST_Data',train=True,download=True,transform=transforms)
trainLoader=torch.utils.data.DataLoader(dataset=trainSet,**trainLoadParams) # ** dic  is a syntax sugar for fitting function parameters with elements of a dic

# Read testset and load with dataloader
testSet=datasets.MNIST(root='./MNIST_Data',train=False,download=True,transform=transforms)
testLoader=torch.utils.data.DataLoader(dataset=testSet,**testLoadParams)

# Define fully connected network with three hidden layers as the model
class FC_Net(nn.Module):
    
    def __init__(self):
        super(FC_Net,self).__init__()
        # Flatten the image to 1d tensor as input. For MNIST the image size is (28,28), so the input size is 28*28 
        self.inputSize=28*28
        # 10 Categories in MNIST, so output size is 10
        self.outputSize=10
        self.hiddenLayerSizes=[256,128,64]
        self.model=nn.Sequential( nn.Linear(self.inputSize,self.hiddenLayerSizes[0]),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hiddenLayerSizes[0],self.hiddenLayerSizes[1]),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hiddenLayerSizes[1],self.hiddenLayerSizes[2]),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hiddenLayerSizes[2],self.outputSize)
                                                    )
    
    def forward(self,flatImg):
        # flatImg is batchSize*inputSize tensor
        # output is batchSize*outputSize tensor
        # Notice that there isn't softmax layer at the end, so the values of dim1 of output is not  probabilities of each catogories yet
        return self.model(flatImg)

# Instantiate the fully connected network model 
model=FC_Net().to(device)

# Instantiate the optimizer with weights of the network and learning rate of the optimizer
optimizer=optim.Adam(params=model.parameters(), lr=learningRate)

# Specify the loss function as cross entropy which is often used for multiclass classification
lossFunc=nn.CrossEntropyLoss()

# Instantiate tensorboard writer
TBwriter=SummaryWriter('./runs/MNIST')

# Training loop
for epoch in range(trainEpochs):
    # Accumulating loss of each epoch
    epochLoss=0 
    for batchIndex, (imgs, labels) in enumerate(trainLoader):
        
        # Move images and labels to GPU is there is one
        imgs, labels=imgs.to(device),labels.to(device)
        
        # A single input  (i.e. a single image in the batch) of  fully connected net is 1d tensor 
        imgs=torch.flatten(imgs,start_dim=1)
        
        # Gradient of left nodes will accumulate, so before backward propagation in each iteration, we need to reset the gradient 
        optimizer.zero_grad()
        
        # Compute the output
        preds=model.forward(imgs)
        
        # probabilities of each class is not normalized in output, but CrossEntropyLoss will do that, it equals LogSoftmax + Negative Log Likelyhood Loss 
        loss=lossFunc(preds,labels)
        
        # Accumulate the loss
        epochLoss+=float(loss)
        
        # Backward propagation
        loss.backward()
        
        # Update weights of network using gradients obtained in backward propagation
        optimizer.step()
        
        # Write the loss of each batch to tensorboard
        TBwriter.add_scalar('Train Loss', float(loss), epoch*len(trainLoader)+batchIndex)
    
    print('Epoch %d, Average loss is: %f' %(epoch,epochLoss/(len(trainLoader))))

# Concatenate predictions and labels of all test batchs, so that we can use sklearn.metrics to compute precision, recall and fbeta score
allPreds,allLabels=torch.LongTensor([]).to(device),torch.LongTensor([])

# Testing Loop
for batchIndex, (imgs, labels) in enumerate(testLoader):
    
    # Only need images to compute predictions, so no need to move labels to GPU
    imgs=imgs.to(device)
    imgs=torch.flatten(imgs,start_dim=1)
    
    # Testing doesn't need to compute gradient
    with torch.no_grad():
        
        # Computer predictions
        preds=model.forward(imgs)
        
        # Normalize the probabilities of all categories with Softmax
        # Shape of preds is batchSize*categoryNum, so softmax the dim1
        preds=nn.functional.softmax(preds,dim=1)
        
        # Pick the category that has the max probability as the prediction
        # Shape of preds is batchSize*categoryNum, so find max in dim1
        top1Prob,top1Indice=torch.max(preds,dim=1)
    
    # Show the prediction result on one test batch if you want
    ''' 
    if batchIndex==0:
        for img,pred in zip(imgs,top1Indice):
            plt.imshow(img.view(28,28).cpu().numpy().squeeze())
            plt.title('prediction: %d' %pred)
            plt.show()
    '''
    
    # Concatenate current batch predictsions and labels with previous ones
    allPreds=torch.cat((allPreds,top1Indice),dim=0)
    allLabels=torch.cat((allLabels,labels),dim=0)


allPreds=allPreds.cpu().numpy()
allLabels=allLabels.numpy()
precion,recall,fbetaScore,_=metrics.precision_recall_fscore_support(y_pred=allPreds,y_true=allLabels,beta=1,average='weighted')
print("{:<25}{}".format('Weighted precision is: ',precion))
print("{:<25}{}".format('Weighted recall is: ',recall))
print("{:<25}{}".format('Weighted fbeta score is: ',fbetaScore))