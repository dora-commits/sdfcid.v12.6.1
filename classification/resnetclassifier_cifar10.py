class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.model(x)
        return x

    def train_model(self, trainloader, testloader, num_epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)

        best_test_loss = float('inf')
        best_model_weights = None
        early_stop_counter = 0
        early_stop_patience = 5

        train_loss_step       = []
        train_acc_step        = []
        validation_loss_step  = []
        validation_acc_step   = []

        for epoch in range(num_epochs):
            train_running_loss    = 0.0
            test_running_loss     = 0.0
            train_correct         = 0.0
            test_correct          = 0.0

            self.train()
            for (images, labels) in tqdm(trainloader):
                images = images.to(device)
                labels = labels.to(device)

                logits = self(images)
                loss = criterion(logits, labels)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                train_running_loss += loss.detach().item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()

            train_loss = train_running_loss / len(trainloader)
            train_acc = train_correct / len(trainloader)
            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f} Train Acc: {train_acc:.6f}')
            train_loss_step.append(train_loss)
            train_acc_step.append(train_acc)

            self.eval()
            with torch.no_grad():
                for (images, labels) in tqdm(testloader):
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = self(images)
                    loss = criterion(logits, labels)

                    test_running_loss += loss.detach().item()
                    predictions = torch.argmax(logits, dim=1)
                    test_correct += (predictions == labels).sum().item()

                test_loss = test_running_loss / len(testloader)
                test_acc = test_correct / len(testloader)
                print(f'Epoch {epoch}: Test Loss: {test_loss:.6f} Test Acc: {test_acc:.6f}')
                validation_loss_step.append(test_loss)
                validation_acc_step.append(test_acc)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model_weights = self.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f'Validation loss has not improved for {early_stop_patience} epochs. Stopping early...')
                        break
                lr_scheduler.step(test_loss)

        self.load_state_dict(best_model_weights)
        self.eval()

        return train_loss_step, train_acc_step, validation_loss_step, validation_acc_step

    def predict_model(self, inputs):
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
        return outputs