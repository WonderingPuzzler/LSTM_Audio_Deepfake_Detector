
# **LSTM Audio Deepfake Detector: Model State Dictionary Included!**

This is part of a project I created for CYBR 4980: Introduction to AI Cybersecurity! My task was to use a type of Artifiicial Intelligence to solve an ongoing problem in the Cybersecurity field. Audio created with the help of Machine Learning (ML), Deep Learning (DL), or with other Artificial Intelligence techniques like Generative Adversarial Networks (GANs), have been the subject of much research for awhile now. However, with the major recent advancements in these technologies, AI-generated audio, commonly referred to as “deepfake” audio, has become a major security issue. With a well-enough cloned audio sample, 
a malicious actor could easily engage in a phishing scam meant to make employees, government officials, or those within any other organization give away privileged information. Through phishing, they could also be used to help a bad actor gain high-level access to organization systems. Perhaps they could even be used to break into a voice-protected security system. In any case, there is the chance of such systems wreaking great havoc on any number of organizations. Therefore, it was the hope of this project to create a method of detecting these AI “deepfakes,” through looking at artifacts created by algorithms called vocoders.
These algorithms are used in most modern audio deepfake technology, and should be able to be detected when looking at Mel-Frequency Cepstral Coefficients or Mel-Spectrograms. 

The LibreSeVoc dataset and inspiration for this project came from this project from Chengzhe Sun, Shan Jia, Shuwei Hou, and Siwei Lyu: https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts. However, I saw that them and many others either used pre-trained models like ResNet, or various other models that I didn't have experience with, namely from-scratch LSTM and CNN-type models. Because of this, I wanted to try my hand at creating a deepfake detector that used no pre-trained models, and worked from scratch from PyTorch's LSTM and CNN model (CNN functionality isn't finished yet, but should hopefully be done by the end of this month).

I was pretty satisfied with the end results! I ended up getting a testing accuracy of 91-94% with the LibreSeVoc Dataset! Here are the Loss/Accuracy Curves and an example Confusion Matrix:

![](Loss_CurvesV9.png)


![](Confusion_MatrixV9_5.png)

I also included a Model State Dictionary in case you want to use this model and test its results out for yourself! It should be called DeepFake_Detector_LSTM_V9.pth 

## Usage

To run locally, you will likely need to run this command in Python:

```bash
# Downloads requirement libraries
pip install -r requirements.txt
```

Then, you will need to download and unzip the LibreSeVoc Dataset (Warning: *very* large): https://zenodo.org/records/15127251

Finally, you will need to change the directory for main.py to scan to whatever directory you put the extract folders in (Warning: it MUST be the directory which shows the folders for each individual class. Otherwise, the program WON'T know what to scan)

However, I've also created a Google Colab Notebook (called LSTM_Deepfake_Detector_Notebook.ipynb) in case you want to run it without taking up any space on your computer (Note: The dataset and model state dictionary *will* need to take up space somewhere. I used Google Drive. Also, I added in commands at the top that can unzip the dataset.)

Again, in both cases, you'll need to change the directory of wherever you decide to put the extracted LibriSeVoc Dataset and Model State Dictionary for the program to function properly.

## Credits

Huge thanks again to Chengzhe Sun, Shan Jia, Shuwei Hou, and Siwei Lyu, who created the dataset I used to train my model, as well as for being the major inspiration behind this project! You can view their paper and original project here: https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.pdf

https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts 

https://www.mdpi.com/1424-8220/25/7/1989

https://arxiv.org/pdf/2308.14970

https://doi.org/10.1016/j.procs.2025.09.097

https://yangcha.github.io/EER-ROC/

https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html

https://www.kaggle.com/code/kvpratama/audio-classification-with-lstm-and-torchaudio

https://medium.com/data-science/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

https://librosa.org/doc/main/generated/librosa.feature.mfcc.html

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

https://medium.com/data-science/t-sne-clearly-explained-d84c537f53a

https://www.nb-data.com/p/breaking-down-the-classification

https://www.mdpi.com/1999-4893/15/5/155

https://link.springer.com/chapter/10.1007/978-3-031-49803-9_11

https://cs230.stanford.edu/projects_fall_2020/reports/55721255.pdf

https://dl.acm.org/doi/fullHtml/10.1145/3643491.3660289

https://dl-acm-org.leo.lib.unomaha.edu/doi/pdf/10.1145/3714458

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a/

https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/#:~:text=The%20softmax%20function%2C%20often%20used%20in%20the,by%20the%20sum%20of%20all%20the%20exponentials

https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/

https://www.geeksforgeeks.org/machine-learning/multiclass-receiver-operating-characteristic-roc-in-scikit-learn/

https://saturncloud.io/blog/using-weights-in-crossentropyloss-and-bceloss-pytorch/

https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

https://www.w3schools.com/python/numpy/numpy_array_join.asp

https://www.geeksforgeeks.org/python/compute-the-mean-standard-deviation-and-variance-of-a-given-numpy-array/

https://numpy.org/doc/stable/reference/generated/numpy.std.html

https://librosa.org/doc/0.11.0/auto_examples/plot_display.html

https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

https://www.analyticsvidhya.com/blog/2023/11/train-test-validation-split/

https://medium.com/@derutycsl/intuitive-understanding-of-mfccs-836d36a1f779

https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html

(Proper APA Credits Coming Soon!)
