import torch
import torchvision
import torchvision.transforms as transforms
import itertools
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Define device as GPU if available
useCuda=torch.cuda.is_available()
device=torch.device('cuda:0' if useCuda else 'cpu')
torch.backends.cudnn.benchmark=True     # If input images have the same size, enable benchmark can improve runtime

# All params
trainEpochs=30

# Trainset loader params
trainLoadParams = { 'batch_size': 16,
                    'shuffle': True,
                    'num_workers':8,
                    'pin_memory':True}

# Testset loader params
testLoadParams = { 'batch_size': 16,
                    'shuffle': False,
                    'num_workers':8,
                     'pin_memory':True}

# Define transformation to transform PIL image to tensor and normalize the input tensors
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Read trainset and load with dataloader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,**trainLoadParams) # ** dic  is a syntax sugar for fitting function parameters with elements of a dic

# Read testset and load with dataloader
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,**testLoadParams)

# Corresponding classes labels of CIFAR10, the order is important and shouldn't be changed
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# Show some of the training images

# Functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
example_data, example_targets = next(iter(trainloader))

# Show images
imshow(torchvision.utils.make_grid(example_data[:4]))

# print labels
print(' '.join('%5s' % classes[example_targets[j]] for j in range(4)))


########################################################################
# Define original AlexNet and my AlexNet variant with FC layers at last replaced by one global average pooling layer


# Original AlexNet
class AlexNet(nn.Module):   
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32*14*14,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num_class),
        )
        self.lossFunc=nn.CrossEntropyLoss()
    
    def forward(self, x):

        x = self.feature(x)
        x = x.view(-1,32*14*14)
        x = self.classifier(x)
        return x


# AlexNet variant with global average pooling
class GlobalPoolAlexNet(nn.Module):   
    def __init__(self, num_class=10):
        super(GlobalPoolAlexNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
            # Last conv layer output channel number should be class number
            nn.Conv2d(32, num_class, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # What GAP does is reducing each channel feature map to a single value, and each channel represent a class
            nn.AvgPool2d(kernel_size=6), 
        )
        self.lossFunc=nn.CrossEntropyLoss()
    
    def forward(self, x):

        x = self.feature(x)
        x = self.classifier(x).squeeze_()
        return x

# Instantiate model
model = GlobalPoolAlexNet().to(device)
# model = AlexNet().to(device)

# Instatiate optimizer
optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.95,weight_decay=1e-3)

# Specify the loss function by getting the loss function from the model
# ( I think it's easier to change and not lead to bug if loss function is included in network model )
lossFunc=model.lossFunc


# Helper function to compute loss of current model on train set and test set
def computeLoss(datasetName):
    accum_loss=0
    num_sample=0
    
    # Specify the reduction as sum so that we can easily compute average loss with accum_loss/num_sample
    lossFunc=nn.CrossEntropyLoss(reduction="sum")
    
    if datasetName=="train":
        loader=trainloader
    else:
        loader=testloader
    
    for i,(inputs,labels) in enumerate(loader):
        if i<=len(testloader)/2: # Only compute the loss on half of train set or test set to save some time
            inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
           
            # Don't need gradient for loss computation
            with torch.no_grad():
                outputs = model.forward(inputs)
                loss = lossFunc(outputs,labels)
                accum_loss+=loss
                num_sample+=len(inputs)
    
    return accum_loss/num_sample   


########################################################################
# Training

# batch loss and batch iteration index container
batch_losses=[]
batch_iters=[]

# train and test loss and epoch iteration index container
train_losses=[computeLoss("train")]
test_losses=[computeLoss("test")]
test_iters=[0]

# Training loop
for epoch in range(trainEpochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, ( inputs, labels) in enumerate(trainloader, 0):
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize + update
        outputs = model.forward(inputs)
        loss = lossFunc(outputs,labels)
        loss.backward()
        optimizer.step()
        
        # Accumulate running loss
        running_loss += loss.item()
        
        # record batch loss at the beginning of each 1000 mini-batches
        if i%1000==0:
            batch_losses.append(loss.item())
            batch_iters.append(epoch*len(trainloader)+i)
            
        # record test loss after each epoch
        if i==len(trainloader)-1:
            train_losses.append(computeLoss("train"))
            test_losses.append(computeLoss("test"))
            test_iters.append(epoch+1)

        # print statistics
        if i==len(trainloader)-1:    # print every 2000 mini-batches
            print("Epoch {}, average training loss: {:.3f}".format(epoch+1,running_loss/len(trainloader)))
            running_loss = 0.0 # Reset running loss

print('Finished Training')


# Plot loss curve of mini-batch, train set and test set
plt.figure(dpi=200)
plt.title('Training loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,2.75])
plt.yticks(np.arange(0,3,0.25))
plt.grid(linestyle='--', linewidth=0.5)
batchLossLine,=plt.plot(np.array(batch_iters)/len(trainloader), batch_losses,label="Batch loss")
trainLossLine, = plt.plot(test_iters, train_losses,label="Train set loss")
testLossLine, = plt.plot(test_iters, test_losses,label="Test set loss")
plt.legend(handles=[batchLossLine,trainLossLine,testLossLine])
plt.show()


########################################################################
# Evaluate accuracy on train set and test set after training

correct = 0
total = 0

# Evaluation on train set
with torch.no_grad():
    for (images, labels) in trainloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on the 60000 train images: {:.2f}%'.format(100 * correct / total))

correct = 0
total = 0

# Counter of each class and confusion matrix
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
cmt = torch.zeros(10,10, dtype=torch.int64)

# Evaluation on test set
with torch.no_grad():
    for ( images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            cmt[labels[i], predicted[i]] += 1
            class_correct[label] += c[i].item()
            class_total[label] += 1


print('Accuracy of the model on the 10000 test images: {:.2f}%'.format(100 * correct / total))

for i in range(10):
    print('Accuracy of {:6} : {:.2f} %'.format(classes[i], 100 * class_correct[i] / class_total[i]))


# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot confusion matrix
plt.figure(dpi=200)
plot_confusion_matrix(cmt.numpy(), classes)

