# Plant_Leaf_Disease_Detection

Plant Leaf disease detection is an computer vision task that involves detection of plants infected with disease from the image of its leaf provided as input. It plays a vital role in improving agriculture as it enables early detection and intervention against plant diseases, thereby improving crop yield and quality. This application provides User-friendly interface for farmers so that they can upload the pictures of plant-leaves to help them detect diseases in crop early and take suitable measures to counter them.    

# MobileNetV2
MobileNetV2 is a convolutional neural network architecture built especially for mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.     

This CNN was chosen as model since it reduces the computations involved, thus making the model lightweight and accessible through mobile devices, also since the MobileNetv2 is pre-trained on ImageNet dataset, it works well on image classification tasks. 

<img width="500" height='300' alt="MobileNetV2" src="https://github.com/AshishViswas/Plant_Leaf_Disease_Detection/assets/130546401/6fb7b1f5-8511-40e6-9289-01b5d65ca6ce">

# PlantVillage_Dataset
The PlantVillage Dataset is a large scale dataset consisting of different plant leaves and their 38 categories of possible infections. It is widely used dataset for research purposes in disease detection in plants.      

the dataset consists of 54,305 images in total spanning 38 classes - ['Corn_(maize)___Common_rust_', 'Cherry_(including_sour)___healthy', 'Apple___Apple_scab', 'Grape___Black_rot', 'Peach___Bacterial_spot', 'Corn_(maize)___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Strawberry___healthy', 'Tomato___Late_blight', 'Blueberry___healthy', 'Tomato___Target_Spot', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Tomato___healthy', 'Apple___healthy', 'Raspberry___healthy', 'Tomato___Bacterial_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Potato___healthy', 'Grape___Esca_(Black_Measles)', 'Cherry_(including_sour)___Powdery_mildew', 'Tomato___Leaf_Mold', 'Soybean___healthy', 'Tomato___Septoria_leaf_spot', 'Apple___Black_rot', 'Orange___Haunglongbing_(Citrus_greening)', 'Grape___healthy', 'Apple___Cedar_apple_rust', 'Tomato___Tomato_mosaic_virus', 'Potato___Early_blight', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Tomato___Early_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Potato___Late_blight', 'Corn_(maize)___Northern_Leaf_Blight', 'Squash___Powdery_mildew', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)']      

# Downloads:  
Download PlantVillage Dataset here:   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data        

The model was trained in an kaggle notebook:   https://www.kaggle.com/code/ashishviswas/plant-leaf-disease-detection    

# Getting Started
To run this project you can download the plant-leaf-disease-detection.ipynb file provided in the repository and the dataset from the download section and can implement the whole process by executing each cell in order in the colab notebook provided in downloads section. 

I choose Kaggle to implement this project because it provides inbuilt GPU accelerator which accelerate the training process, I used GPU T4 x2 to implement this. You can also choose google colab to run this, google colab also provides inbuilt GPU accelerator which fast up the training process much faster that using CPU.     

# Model Training
First, the weights of mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1-0_224_no_top.h5 model have to be loaded. The weights are given in the repository, then freeze first layers and  set last 30 layers as 'trainable', then change the output layer by setting no.of classes as '38' to fine-tune the model on this downstream task.      

To train this model, I used GPU T4 x2 accelerator which accelerated my trained process many times faster than using CPU. In the training process, the training Epochs are set to 5, batch size to 32 and the process went well with higher accuracy and low loss. The model has been trained on PlantVillage dataset provided in the downloads.  

# Saving the trained model
The trained model has been saved : Plant_Leaf_Disease_Detector.h5      

You can find the model weights in this Repository. You can download these weights and can use them to predict on new Images   

# Model Testing
The model has been tested on test generator which consisted of 8146 images which resulted in a accuracy of 96% with a loss of 0.0333 approximately.    

# Metrics Visualization

![training_validation_plot](https://github.com/AshishViswas/Plant_Leaf_Disease_Detection/assets/130546401/67926ce8-3735-4859-8194-23e0f407ea52)

The Above graph visualize the metrics during the training process, it shows Training & Validation Loss and Training & Validation Accuracy with the starting value and ending value. The graphs shows the gradual decrease in the loss function and gradual increase accuracy as shown in the visualization.

# Streamlit App
Streamlit App was used for the implementation of web interface functionality where users can just upload a pic of plant leaf and click the "Classify" button to get the model's prediction.       

The streamlit app can be started by using the command: streamlit run {path_to_main.py} in the terminal

# Requirements
requirements were specified in an requirements.txt file  provided in the repository      

# Predictions 
Model can be used for predictions on new images. Folder 'test_images' has been provided in the repository which contain's few images which can be used for Testing the Model.   

By giving an image as input to streamlit app and then clicking on 'classify' button, we will get the model's prediction which is pretty accurate considering the model is trained with images spanning 38 classes with each class having different no.of images
