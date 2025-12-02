from deep_fake_detector_graphs_and_stats import *

def main() -> None:
    if os.path.exists('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc'):
        detector = DeepFakeDetectorGraphsAndStats(directory='/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc', file_extension='.wav', loss='CrossEntropyLoss', optim='Adam', DL_type='RNN')
        detector.set_batch_size(16) # Make it 16 samples per batch
        detector.set_duration(6) # Set duration to 6 seconds
        detector.set_learning_rate(0.0008) # Set learning rate to 0.001
        detector.setup_data_loaders() # Setup data loaders
        detector.set_optim("Adam") # Set optimizer to Adam
        detector.train_LSTM(num_epochs=100) # Train for 100 epochs
        detector.save_model('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth') # Save the trained model

        detector.plot_training_curves()
        detector.plot_class_counts()
        detector.print_optimizer_loss_architecture('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
        detector.evaluate_model('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
        detector.sns_scatter_plot('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
        detector.create_confusion_matrix('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
        detector.create_classification_report('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
    else:
        print("Dataset directory does not exist.")
 
if __name__ == "__main__":
    main()