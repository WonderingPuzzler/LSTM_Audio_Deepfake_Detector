# Import standard libraries
import time
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

# Import pickle for object serialization
import pickle

# Import standard data science libraries
import random
import numpy as np
import pandas as pd

# Import PyTorch and related libraries
import torch
import torchmetrics, mlxtend
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlxtend.plotting import plot_confusion_matrix

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Import imbalanced-learn for handling imbalanced datasets if we decide to use it
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Import scikit-learn for various utilities
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced visualizations

# Import XGBoost and other classifiers
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Import audio processing libraries
import scipy
import scipy.io.wavfile as wavfile
import scipy.signal

class ClassesDictionarySetUp(Dataset):
    """
    A class to set up and manage dataset properties such as device, directory, classes, class counts,
    data length, transform flag, class indices, and audio file dictionary.

    """
    def __init__(self, directory: str = '', data_length: int = 1048576, transform: bool = False) -> None:
        """
        Initialize the dataset setup with optional parameters for directory, data length, and transform flag.
        
        Parameters:
            directory (str): The directory path of the dataset. Defaults to an empty string.
            data_length (int): The fixed length for audio data samples. Defaults to 1048576.
            transform (bool): A flag indicating whether to apply transformations to audio data. Defaults to False

        Returns:
            None
        """

        super().__init__() # Initialize the parent Dataset class

        # Initialize private attributes with default values
        self.__device = None  
        self.__directory = None
        self.__classes: List[str] = []
        self.__classes_index: Dict[str, int] = {}
        self.__class_counts: Dict[str, int] = {}
        self.__file_dictionary = {}
        self.__data_length = 0
        self.__transform = False


        # Choose device: prefer CUDA when available
        # Perform setup actions (these methods set the private attributes internally)
        self.setupDevice()

        # Prefer provided directory when calling set_directory
        if directory:
            self.set_directory(directory)
        else:
            self.set_directory()

        # Ensure classes and counts are initialized
        self.set_classes()
        self.set_class_counts()

        # Set data length and transform
        self.set_data_length(data_length)
        self.set_transform(transform)

        # Initialize class index mapping and create classes and counts
        self.create_classes_and_class_counts()

        # Set up file dictionary by scanning the dataset directory
        self.setup_file_dictionary('.wav')

    
    def get_device(self) -> torch.device:
        """ 
        Get the device used for PyTorch computations.

        Parameters:
            None

        Returns:
            torch.device: The device (CPU or GPU). If not set, defaults to CPU.
        """

        # Ensure device is set and available
        if not self.__device or not torch.cuda.is_available(): 
            self.__device = torch.device("cpu")

        return self.__device 
    
    def set_device(self, device: torch.device) -> None:
        """ 
        Set the device for PyTorch computations.

        Parameters:
            device (torch.device): The device to set (CPU or GPU). If None or CUDA not available, defaults to CPU.

        Returns:
            None
        """

        # Ensure device is valid and available
        if device == None or not torch.cuda.is_available():
            self.__device = torch.device("cpu")

        self.__device = device

    def get_directory(self) -> str:
        """  
        Get the directory path of the dataset.

        Parameters:
            None

        Returns:
            str: The directory path. If empty, returns default path.
        """

        # Ensure directory is set
        if not self.__directory or not os.path.isdir(self.__directory):
            self.__directory = '/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc'

        return self.__directory
    
    def set_directory(self, directory: str = '') -> None:
        """ 
        Set the directory path of the dataset.

        Parameters:
            directory (str): The directory path to set. If empty, uses default path.

        Returns:
            None
        """

        # Ensure directory is valid by checking if it exists and is a directory
        if not directory or not os.path.isdir(directory):
            directory = '/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc'

        self.__directory = directory

    def get_classes(self) -> List[str]:
        """ 
        Get the list of classes in the dataset.

        Parameters:
            None

        Returns:
            List[str]: The list of class names. If empty, returns ['None'].
        """

        # Ensure classes at least has a default value
        if not self.__classes:
            self.__classes = ['None']

        return self.__classes
    
    def set_classes(self, classes: List[str] = []) -> None:
        """ 
        Set the list of classes in the dataset by scanning the directory structure.

        Parameters:
            classes (List[str], optional): The list of class names to set. If empty, initializes to ['None'].

        Returns:
            None
        """

        # Ensure classes is a list of strings
        if classes is not None:
            self.__classes = classes
        elif not self.__classes:
            self.__classes = ['None']

    def get_class_counts(self) -> Dict[str, int]:
        """ 
        Get the counts of samples per class in the dataset.

        Parameters:
            None

        Returns:
            Dict[str, int]: A dictionary with class names as keys and their respective counts as values. 
            If empty, returns {'None': 0}.
        """

        # Ensure class counts at least has a default value
        if not self.__class_counts:
            self.__class_counts = {'None': 0}

        return self.__class_counts

    def set_class_counts(self, class_counts: Dict[str, int] = {}) -> None:
        """ 
        Set the counts of samples per class in the dataset by scanning the directory structure.

        Parameters:
            class_counts (Dict[str, int], optional): A dictionary with class names as keys and their respective counts as values. 
            If empty, initializes to {'None': 0}.

        Returns:
            None
        """

        # Ensure class counts is a dictionary with a tuple of (string, integer)
        if class_counts is not None:
            self.__class_counts = class_counts
        elif not self.__class_counts:
            self.__class_counts = {'None': 0}

    def get_data_length(self) -> int:
        """ 
        Get the fixed length for audio data samples.

        Parameters:
            None
        Returns:
            int: The fixed length for audio data samples.
        """

        return self.__data_length
    
    def set_data_length(self, data_length: int) -> None:
        """ 
        Set the fixed length for audio data samples.

        Parameters:
            data_length (int): The fixed length to set for audio data samples.

        Returns:
            None
        """

        # If data_length is positive, set it; otherwise, raise an error
        if data_length > 0:
            self.__data_length = data_length
        else:
            raise ValueError("Data length must be a positive integer.")
        
    def get_transform(self) -> bool:
        """ 
        Get the transform flag indicating whether to apply transformations to audio data.

        Parameters:
            None

        Returns:
            bool: The transform flag.
        """

        return self.__transform
    
    def set_transform(self, transform: bool) -> None:
        """ 
        Set the transform flag indicating whether to apply transformations to audio data.

        Parameters:
            transform (bool): The transform flag to set.

        Returns:
            None
        """

        # Ensure transform is a boolean and raise an error if not
        if isinstance(transform, bool):
            self.__transform = transform
        else:
            raise ValueError("Transform must be a boolean value.")


    def get_classes_index(self) -> Dict[str, int]:
        """ 
        Get the mapping of class names to their respective indices.

        Parameters:
            None

        Returns:
            Dict[str, int]: A dictionary mapping class names to their respective indices. 
            If empty, returns a dictionary with 'None' mapped to 0.
        """

        # Ensure classes index at least has a default value of {'None': 0}
        if not self.__classes_index:
            self.__classes_index = {'None': 0}

        return self.__classes_index
    
    def set_classes_index(self, classes_index: Dict[str, int] = {}) -> None:
        """ 
        Set the mapping of class names to their respective indices.

        Parameters:
            classes_index (Dict[str, int], optional): A dictionary mapping class names to their respective indices. 
            If empty, initializes to a dictionary with 'None' mapped to 0.

        Returns:
            None
        """

        # Ensure classes index is at least a dictionary with 'None' mapped to 0
        if classes_index is not None:
            self.__classes_index = classes_index
        elif not self.__classes_index:
            self.__classes_index = {'None': 0}


    def get_file_dictionary(self) -> Dict[int, List[str]]:
        """ 
        Get the dictionary of file paths along with their corresponding class indices.

        Parameters:
            None

        Returns:
            Dict[int, List[str]]: A dictionary mapping class indices to lists of file paths.
            If empty, returns {0: ['None']}.
        """

        # Return the file dictionary mapping class_index -> list[str].
        # If empty, return an empty dict.
        if not self.__file_dictionary:
            return {0: ['None']}
        else:
            return self.__file_dictionary

    def set_file_dictionary(self, file_dictionary: Dict[int, List[str]]) -> None:
        """ 
        Set the dictionary of file paths along with their corresponding class indices.

        Parameters:
            file_dictionary (Dict[int, List[str]]): A dictionary mapping class indices to lists of file paths.

        Returns:
            None
        """

        # Accept a dictionary mapping class_index -> list[str]
        if file_dictionary is not None:
            self.__file_dictionary = file_dictionary
        else:
            self.__file_dictionary = {0: ['None']}

    def setupDevice(self) -> None:
        """ 
        Setup device for PyTorch computations by checking for CUDA availability. 
        If CUDA is available, it sets the device to GPU; otherwise, it defaults to CPU.

        Parameters:
            None

        Returns:
            None        
        """

        print(" =============================== Device Setup =============================== \n")
        self.set_device(torch.device("cuda")) # Try to set to CUDA 
        print(f"Using device: {self.get_device()}") # Print the device being used

        # Check if CUDA is available and print relevant information (get_device will automatically default to CPU if not)
        if self.get_device().type == "cuda":
            try:
                # Print basic GPU info
                print("GPU:", torch.cuda.get_device_name(0))
                print("CUDA available:", torch.cuda.is_available())
            except Exception:
                pass

        # Otherwise, default to CPU
        else:
            print("CUDA not available, using CPU.")
        print ("\n ----------------------------- Setup Complete ------------------------------ \n")


    def create_classes_and_class_counts(self) -> None:
        """ 
        This function sets up the logic needed to not only find the classes in the dataset directory 
        and find their counts, but also to create a mapping from class names to indices.
        
        Parameters:
            None
            
        Returns:
            None
        """

        print(" \n=============================== Creating Classes, Class Counts, and Class Indices =============================== \n")

        # Go through the dataset directory, finding the name of each file within the directory, and ensure it is a directory
        # If it is, add it to the classes list
        classes = sorted([d for d in os.listdir(self.get_directory()) if os.path.isdir(os.path.join(self.get_directory(), d))])

        # Set the classes found and print them
        self.set_classes(classes)
        print (" Classes found: ", list(self.get_classes()))

        # Now, create the class counts and class indices
        class_counts = {}
        classes_index = {}

        # For each class found, count the number of files in its directory and assign an index
        for index, class_name in enumerate(self.get_classes()):
            # Get the directory path for the current class by joining the base directory with the class name
            class_dir = os.path.join(self.get_directory(), class_name)

            # The class count is simply the number of files in that directory which can be found by getting a length of the list of files in a directory
            class_counts[class_name] = len(os.listdir(class_dir))

            # The index is simply the enumeration order (0, 1, 2, ...)
            classes_index[class_name] = index

        # Store the class counts and indices and print them
        self.set_class_counts(class_counts)
        self.set_classes_index(classes_index)
        print (" Class counts: ", self.get_class_counts())
        print (" Class indices: ", self.get_classes_index())

        print (" \n----------------------------- Creation of Classes, Class Counts, and Class Indices Complete ------------------------------ \n")

    def setup_file_dictionary(self, file_type) -> None:
        """ 
        This function sets up the file lists by scanning the dataset directory 
        and pairing each file path with its corresponding class index.
        
        Parameters:
            file_type (str): The file extension to filter files (e.g., '.wav').
            
        Returns:
            None
        """

        print(" \n=============================== Setting up File Dictionary =============================== \n")

        # Placeholder to build the file dictionary
        file_dict: Dict[int, List[str]] = {}

        # Build a mapping from class index -> list of full file paths for that class
        for class_name in self.get_classes():

            # Get the directory path for the current class by joining the base directory with the class name
            class_dir = os.path.join(self.get_directory(), class_name)

            # Collect wav filenames for this class and sort them by going through the directory (f for f in os.listdir... )
            files = sorted([f for f in os.listdir(class_dir) if f.endswith(file_type)])

            # Get the full paths by joining the class directory with each filename (needed if we want to load the actual data later)
            full_paths = [os.path.join(class_dir, f) for f in files]

            # Get the class index from the class name
            class_index = self.get_classes_index().get(class_name, None)
            
            if class_index is None:
                # Fallback: use enumeration order if index missing
                class_index = len(file_dict)

            # Store the list of full paths under the class index
            file_dict[class_index] = full_paths

            # Print files for this class immediately with only basenames for readability
            basenames = [os.path.basename(p) for p in full_paths]
            print(f"\n Class {class_name}: Files: {basenames}\n\n")

        # Store the dictionary
        self.set_file_dictionary(file_dict)

        # Print total number of audio files found
        total_files = sum(len(v) for v in file_dict.values())
        print(f" Total audio files found: {total_files} ")
        print (" \n----------------------------- File Dictionary Setup Complete ------------------------------ \n")
        

    def __getitem__(self, index: int) -> Any:
        """
        This function will get a specific malware file, 
        read data_length bytes from it, and return the data along with its class index.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:   
            Any: A tuple containing the audio data and its corresponding class index.
        """


        return None  # Placeholder for actual implementation



    # TODO: Might eventually put into a subclass of PyTorchDeepFakeDetector called DeepFakeDetectorGraphingAndStats
    def plot_class_counts(self) -> None:
        """ 
        Plot the distribution of samples per class in the dataset.

        Parameters:
            None

        Returns:
            None
        """

        print(" \n=============================== Sample Count Diagramming =============================== \n")

        class_counts = self.get_class_counts()

        # Plotting the class distribution
        plt.figure(figsize=(10, 6))
        plt.bar(list(class_counts.keys()), list(class_counts.values()))
        plt.xlabel("Classes")
        plt.ylabel("Number of Samples")
        plt.title("Class Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(" \n---------------------------- Sample Count Diagramming Complete ---------------------------- \n")


class PyTorchDeepFakeDetector(ClassesDictionarySetUp, nn.Module):
    def __init__(self, directory: str = '', data_length: int = 1048576, transform: bool = False):
        super(PyTorchDeepFakeDetector, self).__init__(directory, data_length, transform)

    def forward(self, x):
        # Forward pass logic here
        pass

def main() -> None:
    if os.path.exists('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc'):
        detector = PyTorchDeepFakeDetector(directory='/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc')
        detector.plot_class_counts()
    else:
        print("Dataset directory does not exist.")

if __name__ == "__main__":
    main()