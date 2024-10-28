import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models import AlexNet_Weights


print(torch.cuda.is_available() )
device = torch.device("cuda")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False


model.classifier[6] = nn.Linear(4096, 10) 


model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.825)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.25)


def train_model(num_epochs):
    model.train()
    train_loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
           
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f'{name}: {param.grad}')
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return train_loss_history


def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy


def plot_loss_history(loss_history):
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


num_epochs = 30
loss_history = train_model(num_epochs)
test_model()
plot_loss_history(loss_history)


def visualize_prediction(image_index):
    image, label = test_dataset[image_index]
    model.eval()
    with torch.no_grad():

        image = image.to(device).unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        
    plt.imshow(image.cpu().squeeze().permute(1, 2, 0), cmap='gray')
    plt.title(f'Predicted: {predicted.item()}, True: {label}')
    plt.show()


visualize_prediction(0)
