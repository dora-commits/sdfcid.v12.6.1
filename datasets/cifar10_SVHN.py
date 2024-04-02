# TODO : Load dataset

# Normalize in range [-1, 1]
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])

# Normalize in range [0, 1] Default
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])

if (args['data_name'] == 'CIFAR10'):
    train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
else:
    train_dataset = torchvision.datasets.SVHN(root='./', split='train', download=True, transform=transform)
    test_dataset  = torchvision.datasets.SVHN(root='./', split='test', download=True, transform=transform)


dataset = ConcatDataset([train_dataset, test_dataset])
temporatory = DataLoader(dataset, batch_size=(len(train_dataset) + len(test_dataset)), shuffle=True)
dec_x_a, dec_y_a = next(iter(temporatory))

foldsX, foldsY = cross_validation_5folds(dec_x_a, dec_y_a)
del dec_x_a
del dec_y_a
     