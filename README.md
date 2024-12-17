CNN for Image Classification - Cats vs Dogs
This project demonstrates how to use a Convolutional Neural Network (CNN) for image classification. The task is to classify images of cats and dogs into two categories: Cat and Dog.
Project Overview
In this project, we train a CNN to classify images of cats and dogs. The model is built using Keras (with TensorFlow as the backend) and performs the following tasks:
Data Preprocessing: The images are resized and normalized for model input.
Model Architecture: A CNN model with convolutional layers, max-pooling layers, and a fully connected output layer.

Training: The model is trained using a labeled dataset of cat and dog images.
Prediction: The trained model is used to predict whether a given image is a cat or a dog.

Project Structure
CNN for Image Classification.ipynb: The Jupyter notebook that contains the code for building, training, and evaluating the CNN model.
README.md: This file that provides project details and setup instructions.

Dataset
The dataset consists of 8,000 images (4,000 cats and 4,000 dogs), split into training and test sets. The images are labeled and categorized into cats and dogs. The dataset can be downloaded from the link below:
Dataset Link: https://drive.google.com/drive/u/2/folders/1FaqPc-iGUyE5sqg_uiA7m7P09T1b5pQG

Dataset Structure:
Training Set: 4,000 images of cats and 4,000 images of dogs, split into subfolders cats/ and dogs/.
Test Set: 1,000 images of cats and 1,000 images of dogs, similarly structured into cats/ and dogs/ folders.

Single Prediction: This folder should contain images you want to classify after training the model.

Setup Instructions
1. Clone the Repository or Download the Files
To get started, you can either clone the repository or download the notebook and dataset directly.
repository link: https://github.com/Varshitha0612/CNN-for-Image-Classification

3. Install Dependencies
Make sure you have Python 3.7+ and the required libraries installed. You can install the dependencies using pip:
bash
Copy code
pip install -r requirements.txt

4. Prepare the Dataset
Download the dataset from the cloud storage link above and place it in the appropriate folder structure:
markdown
Copy code
dataset/
  ├── training_set/
      ├── cats/
      ├── dogs/
  ├── test_set/
      ├── cats/
      ├── dogs/
  ├── single_prediction/

5. Run the Jupyter Notebook
Open the Jupyter notebook (CNN for Image Classification.ipynb) in your preferred environment (e.g., Jupyter Notebook or JupyterLab). You can run it by navigating to the folder containing the notebook and typing:
bash
Copy code
jupyter notebook CNN\ for\ Image\ Classification.ipynb

6. Train and Predict
Train the model by running the code cells in the notebook.
Use the predict_image() function to classify a new image of a cat or a dog.
Model Evaluation
After training, the model will be evaluated using the test set, which includes 1,000 images of cats and 1,000 images of dogs. The model will classify each image as either a cat or a dog.

6.Medium Article
You can read the detailed tutorial and explanation of this project in my Medium article:
Medium Post link: https://medium.com/@varshitha0612/building-a-convolutional-neural-network-cnn-to-classify-cats-vs-dogs-96ec8e88b276

Medium Post link: https://medium.com/@varshitha0612/building-a-convolutional-neural-network-cnn-to-classify-cats-vs-dogs-96ec8e88b276

Summary of Updates:
Added Medium Link: Medium Post link: https://medium.com/@varshitha0612/building-a-convolutional-neural-network-cnn-to-classify-cats-vs-dogs-96ec8e88b276
Dataset Link: https://drive.google.com/drive/u/2/folders/1FaqPc-iGUyE5sqg_uiA7m7P09T1b5pQG
