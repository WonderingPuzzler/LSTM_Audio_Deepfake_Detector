from batch_loss_and_optimization import *

class PyTorchDeepFakeDetectorLSTM(BatchLossAndOptimization):
    """ 
    Creates, trains, and tests a DeepFake Detector using an LSTM architecture.
    Inherits from BatchLossAndOptimization, nn.Module, Dataset, and ClassesFilesDictionarySetUp.
    """
    def __init__(self, directory: str = '', optim: str = 'Adam', loss: str = 'CrossEntropyLoss', file_extension: str = '.wav', DL_type: str = 'RNN') -> None:
        """
        Initialize the PyTorchDeepFakeDetectorLSTM class.
        Sets up the LSTM model architecture and initializes the parent class.
        Also sends the parent the directory, file extension, loss function, optimizer, and DL type.

        Parameters:
            directory (str): Directory containing the dataset.
            optim (str): Optimizer to use for training (default: 'Adam').
            loss (str): Loss function to use for training (default: 'CrossEntropyLoss').
            file_extension (str): File extension of audio files (default: '.wav').
            DL_type (str): Type of deep learning model (default: 'RNN').

        Returns:
            None
        """


        super(PyTorchDeepFakeDetectorLSTM, self).__init__(directory, file_extension, loss, optim, DL_type)

        self.LSTM_model()
        
        # Move model to device (GPU if available)
        self.to(self.get_device())

    def LSTM_model(self) -> nn.Module:
        """
        Define our Long Short-Term Memory (LSTM) model architecture for DeepFake detection.
        Returns the constructed LSTM model.
        e.g. the LSTM model architecture consists of:
        - Two LSTM layers with dropout
        - Fully connected layer for final classification


        Parameters:
            None
        
        Returns:
            nn.Module: The LSTM model.
        """

        # First LSTM layer for initial feature extraction
        self.lstm_1: nn.LSTM = nn.LSTM(
            input_size=self.get_n_mfcc(),  # Number of MFCC features
            hidden_size=256,                # Number of features in hidden state
            num_layers=1,                   # Number of stacked LSTM layers
            batch_first=True,                # Input/output tensors have shape (batch, seq, feature)
            bidirectional=True,              # Bidirectional LSTM for better context
        )

        # By doing a 0.5 dropout, we make 50% of the neurons inactive during each training iteration
        self.dropout1: nn.Dropout = nn.Dropout(0.5)  

        # Second LSTM layer for deeper feature extraction
        self.lstm_2: nn.LSTM = nn.LSTM(
            input_size=512,                 # 256 * 2 (bidirectional)
            hidden_size=1024,                # Number of features in hidden state
            num_layers=1,                   # Number of stacked LSTM layers
            batch_first=True,                # Input/output tensors have shape (batch, seq, feature)
            bidirectional=True,              # Bidirectional LSTM for better context
        )

        # By doing a 0.5 dropout, we make 50% of the neurons inactive during each training iteration
        self.dropout2: nn.Dropout = nn.Dropout(0.5)  

        # Final fully connected layer for classification
        self.fc: nn.Linear = nn.Linear(2048, len(self.get_classes()))  # 1024 * 2 (bidirectional)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.
        Returns the output logits for each class.
        e.g. if we have 2 classes (Real, Fake), output shape is (batch_size, 2)
        If we have 7 classes (real, various DeepFake methods), output shape is (batch_size, 7)

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, n_mfcc).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """

        out: torch.Tensor = x

        # Pass through first LSTM layer
        out, _ = self.lstm_1(x)
        out = self.dropout1(out)

        # Pass through second LSTM layer
        out, _ = self.lstm_2(out)
        out = self.dropout2(out)

        # Take the output from the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through the final fully connected layer
        out = self.fc(out)  # Shape: (batch_size, num_classes)

        return out
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the LSTM model before the final classification layer.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, n_mfcc).
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, hidden_size).
        """

        out: torch.Tensor = x

        # Pass through first LSTM layer
        out, _ = self.lstm_1(x)
        out = self.dropout1(out)

        # Pass through second LSTM layer
        out, _ = self.lstm_2(out)
        out = self.dropout2(out)

        # Take the output from the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        return out



    def train_LSTM(self, num_epochs: int = 20) -> None:
        """
        Train the LSTM using our pre-chosen optimizer and loss function:

        Parameters:
            num_epochs (int): Number of training epochs.
            optimizer_name (str): Name of the optimizer to use.
            loss_name (str): Name of the loss function to use.

        Returns:
            None
        """

        # Ensure DataLoaders are set up
        if self.get_training_loader() is None or self.get_validation_loader() is None:
            print("DataLoaders were not set up. Calling setup_data_loaders() before training.")
            self.setup_data_loaders()

            self.get_training_loader()
            self.get_validation_loader()

        # Initialize lists to track metrics
        self.train_loss_list: list[float] = []
        self.train_acc_list: list[torch.Tensor] = []
        self.val_loss_list: list[float] = []
        self.val_acc_list: list[torch.Tensor] = []

         # Set up optimizer and loss function
        optimizer_class: type = self.get_optim()
        optimizer: torch.optim.Optimizer = optimizer_class(params=self.parameters(), lr=self.get_learning_rate())
        loss: torch.nn.Module = self.get_loss()

        print("\n=============================== Starting LSTM Training ===============================\n")

        # Get DataLoaders
        training_loader: torch.utils.data.DataLoader = self.get_training_loader()
        validation_loader: torch.utils.data.DataLoader = self.get_validation_loader()

        # Check if DataLoaders are available
        if training_loader is None or validation_loader is None:
            raise ValueError("DataLoaders are not available.")

        # Training loop
        for epoch in range(num_epochs):

            self.train()  # Set model to training mode
            epoch_loss: float = 0.0 # Accumulate loss over the epoch
            num_batches: int = 0 # Number of batches processed in the epoch

            train_correct: torch.Tensor = torch.tensor(0, dtype=torch.int32) # Number of correct predictions in training
            train_total: int = 0 # Total number of samples in training
            
            # Training loop with tqdm progress bar
            train_pbar: tqdm = tqdm(training_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=False)

            for batch_idx, (data, labels) in enumerate(train_pbar):

                # Move data to device
                data: torch.Tensor = data.to(self.get_device())
                labels: torch.Tensor = labels.to(self.get_device())
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs: torch.Tensor = self.forward(data)
                
                # Compute loss
                batch_loss: torch.Tensor = loss(outputs, labels)
                
                # Backward pass
                batch_loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Accumulate loss and accuracy
                epoch_loss += batch_loss.item()
                num_batches += 1
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, dim=1)

                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar with current loss
                train_pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
            
            # Calculate average loss and accuracy for the epoch
            avg_loss: float = epoch_loss / num_batches

            train_accuracy: torch.Tensor = 100 * train_correct / train_total
            
            # Validation phase
            self.eval()  # Set model to evaluation mode
            val_loss: float = 0.0 # Accumulate validation loss
            val_correct: torch.Tensor = torch.tensor(0, dtype=torch.int32) # Number of correct predictions in validation
            val_total: int = 0 # Total number of samples in validation
            
            # Track per-class accuracy for validation
            num_classes: int = len(self.get_classes())
            val_class_correct: list[int] = [0] * num_classes
            val_class_total: list[int] = [0] * num_classes
            
            # Validation loop with tqdm progress bar
            val_pbar: tqdm = tqdm(validation_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=False)
            with torch.no_grad():
                for data, labels in val_pbar:

                    # Move data to device
                    data: torch.Tensor = data.to(self.get_device())
                    labels: torch.Tensor = labels.to(self.get_device())
                    
                    # Forward pass
                    outputs: torch.Tensor = self.forward(data)
                    
                    # Compute loss
                    batch_loss: torch.Tensor = loss(outputs, labels)

                    # Accumulate validation loss
                    val_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, dim=1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Track per-class accuracy
                    for i in range(len(labels)):
                        # Get the true label
                        label: int = int(labels[i].item())

                        # Update total count for this class
                        val_class_total[label] += 1

                        # Update correct count for this class if the prediction is correct
                        if predicted[i] == labels[i]:
                            val_class_correct[label] += 1
                    
                    # Update progress bar
                    val_pbar.set_postfix({'val_loss': f'{batch_loss.item():.4f}'})
            
            # Calculate average validation loss and accuracy
            avg_val_loss: float = val_loss / len(validation_loader)
            val_accuracy: torch.Tensor = 100 * val_correct / val_total

            # Store metrics for plotting
            self.train_loss_list.append(avg_loss)
            self.train_acc_list.append(train_accuracy)
            self.val_loss_list.append(avg_val_loss)
            self.val_acc_list.append(val_accuracy)

            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Print per-class validation accuracy
            print(f"  Per-Class Validation Accuracy:")

            class_names: list[str] = self.get_classes()
            # Iterate over each class to print accuracy
            for i, class_name in enumerate(class_names):

                # Calculate and print accuracy for this class
                if val_class_total[i] > 0:

                    # Calculate accuracy for this class
                    class_acc: float = 100 * val_class_correct[i] / val_class_total[i]
                    # Print accuracy for this class
                    print(f"    {class_name}: {class_acc:.2f}% ({val_class_correct[i]}/{val_class_total[i]})")

                # Handle case with no samples for this class
                else:
                    print(f"    {class_name}: No samples in validation set")
            
            # Early stopping if validation loss is below threshold
            if avg_val_loss < 0.40:
                print(f"\n*** Early stopping triggered: Validation loss {avg_val_loss:.4f} < 0.40 ***")
                print(f"*** Training completed at epoch {epoch + 1}/{num_epochs} ***\n")
                break
            
        print("\n---------------------------- LSTM Training Complete ----------------------------\n")


    def evaluate_model(self, past_model: str = 'None') -> None:
        """
        Evaluate the trained model on the test set and report accuracy, loss, and per-class metrics.

        Parameters:
            past_model (str): Path to a saved model to load before evaluation. Defaults to 'None'.

        Returns:
            None
        """

        print("\n=============================== Evaluating Model on Test Set ===============================\n")

        # Load past model if specified
        if past_model != 'None':
            print(f"Loading model from: {past_model}")
            self.load_state_dict(torch.load(past_model, map_location=self.get_device()), strict=False)
            print("Model loaded successfully.\n")

        # Get testing DataLoader
        testing_loader: torch.utils.data.DataLoader = self.get_testing_loader()
        
        # Ensure testing DataLoader is available
        if testing_loader is None:
            print("Testing DataLoader not found. Setting up data loaders now.")
            self.setup_data_loaders()
            testing_loader = self.get_testing_loader()

        self.eval()  # Set model to evaluation mode
        test_loss: float = 0.0
        test_correct: torch.Tensor = torch.tensor(0, dtype=torch.int32)
        test_total: int = 0

        # Track per-class accuracy
        num_classes: int = len(self.get_classes())
        class_correct: list[int] = [0] * num_classes
        class_total: list[int] = [0] * num_classes

        # Get loss function
        loss_fn: torch.nn.Module = self.get_loss()

        # Final check if testing DataLoader is available
        if testing_loader is None:
            raise ValueError("Testing DataLoader is not available.")

        with torch.no_grad():  # No gradient computation during evaluation
            test_pbar: tqdm = tqdm(testing_loader, desc=f"[Testing]", leave=False)
            for data, labels in test_pbar:

                # Move data to device
                data: torch.Tensor = data.to(self.get_device())
                labels: torch.Tensor = labels.to(self.get_device())
                
                # Forward pass
                outputs: torch.Tensor = self(data)
                batch_loss: torch.Tensor = loss_fn(outputs, labels)
                test_loss += batch_loss.item()
                
                # Calculate predictions
                _, predicted = torch.max(outputs, dim=1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Per-class accuracy tracking
                for label, prediction in zip(labels, predicted):
                    # Update total count for this class
                    class_total[label] += 1

                    # Update correct count for this class if prediction is correct
                    if label == prediction:
                        class_correct[label] += 1
                
                # Update progress bar
                test_pbar.set_postfix({'test_loss': f'{batch_loss.item():.4f}'})
        
        # Calculate overall metrics
        avg_test_loss: float = test_loss / len(testing_loader)
        test_accuracy: torch.Tensor = 100 * test_correct / test_total
        
        # Print overall test results
        print(f"\nTest Results:")
        print(f"  Average Loss: {avg_test_loss:.4f}")
        print(f"  Overall Accuracy: {test_accuracy:.2f}% ({test_correct}/{test_total})")
        
        # Print per-class accuracy
        print(f"\nPer-Class Accuracy:")

        # Get class names
        class_names: list[str] = self.get_classes()

        # Iterate over each class to print accuracy
        for i, class_name in enumerate(class_names):

            # Calculate and print accuracy for this class
            if class_total[i] > 0:

                # Calculate accuracy for this class
                class_acc: float = 100 * class_correct[i] / class_total[i]

                # Print accuracy for this class
                print(f"  {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

            # Handle case with no samples for this class
            else:
                print(f"  {class_name}: No samples in test set")
        
        print("\n---------------------------- Model Evaluation Complete ----------------------------\n")