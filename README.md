A simple project of a digit recognition problem. The dataset has been imported from the keras datasets module.  
# Dataset
The training set contains 60000 images of digits, each image is depicted as a 2D, 28x28 array with values in range from 0 to 255.  
The test set contains 10000 images of digits.  
# CNN structure
The CNN consists of 2 sequential sets of layers (Convolutional 2D -> Pooling -> Dropout), Flattening layer and 2 Dense layers.  
# Accuracy and loss plots (25 epochs)  
![image](https://user-images.githubusercontent.com/72389636/124603456-3dd1ee80-de6a-11eb-86bf-7c8d4965a161.png)
![image](https://user-images.githubusercontent.com/72389636/124603549-5215eb80-de6a-11eb-86f3-fbe18eca4907.png)  
# Accuracy and loss plots (10 epochs)  
![image](https://user-images.githubusercontent.com/72389636/124926026-5cb4ba00-dffd-11eb-8c0b-5e9bffda1bfc.png)  
![image](https://user-images.githubusercontent.com/72389636/124925985-5292bb80-dffd-11eb-984c-f42688b736cd.png)  
# Requirements
```tensorflow==2.5.0```  
```matplotlib==3.4.1```  
```numpy==1.19.5```
