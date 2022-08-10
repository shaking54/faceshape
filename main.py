from networks.models import initialize_model
from networks.train import *
from networks.ViT import *
from processingImage.emb import load_data
import argparse


def DeepNetworks(args):

    data_transforms = {
    'training_set': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'testing_set': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    data_dir = args.datadir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['training_set', 'testing_set']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['training_set', 'testing_set']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['training_set', 'testing_set']}
    
    class_names = image_datasets['training_set'].classes

    criterion = nn.CrossEntropyLoss()

    model_ft, input_size = initialize_model(model_name="resnet", num_classes=5, feature_extract=True, use_pretrained=True)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    num_epochs = args.epochs
    model_name = args.model_name
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    

def VisionTransformer(args):

    X_train, y_train,train_path = load_data('/data/FaceShape Dataset/training_set')
    X_test, y_test, test_path = load_data('/data/FaceShape Dataset/testing_set')

    train_transforms = T.Compose([
        T.Resize((224,224)),
        #T.RandomResizedCrop((224,224)  ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                                        
    ])

    test_transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                                        
    ])

    train = CustomImageDataset(train_path, y_train, X_train, transform=train_transforms) 
    test = CustomImageDataset(test_path, y_test, X_test, transform=test_transforms) 

    train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=16, shuffle=True)

    dataloaders ={
                "training_set": train_dataloader,
                "testing_set": test_dataloader
            }
    num_epochs = args.epochs
    model_ft = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    model_ft.to(device)
    
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--datadir', default="data/FaceShape Dataset", 
                        help='data path')
    parser.add_argument('-m','--model_name', default="resnet",
                        help='model name')
    parser.add_argument('-e','--epochs', default=10,
                        help='number of epochs training')

    args = parser.parse_args()

    if args.model_name == "transformers":
        VisionTransformer(args)
    else:
        DeepNetworks(args)

    