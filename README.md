# imageCap
Repository the team challenge of image captioning in the lecture Competitive Problem Solving with Deep Learning

## Installation

```bash  
pip install -r requirements.txt
get fastText repo and install locally with pip
```

## Local Folder Structure

├── data
│   ├── annotations                 <- annotations jsons
│   └── preprocessed                <- results from data_gen.prebuild_training_files()
│   └── train2014                   <- training set folder
│   └── val2014                     <- validation set folder
│   └── resnet152_weights_tf.h5     <- pretrained resnet weights
│    
├── data_generator                  <- Data Generator for train and val data
│   
│
├── embeddings                      <- fastText embeddings
│   ├── wiki.en.bin             
│   └── wiki.en.vec
│             
├── layers                          <- 
│             
├── models                          <-
│   ├── image_captioning_model.py             
│   └── language_model.py
│   └── resnet_152_image_encoder.py