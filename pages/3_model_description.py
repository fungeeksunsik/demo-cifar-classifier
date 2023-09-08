import streamlit as st

st.header("MLP model architecture")
st.markdown("""
    ```
    Model: "cifar10_classifier_mlp"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 32, 32, 3)]       0         
                                                                     
     grayscale_convert_layer (La  (None, 32, 32, 1)        0         
     mbda)                                                           
                                                                     
     flatten_layer (Flatten)     (None, 1024)              0         
                                                                     
     dense_layer_1 (Dense)       (None, 512)               524800    
                                                                     
     dense_layer_2 (Dense)       (None, 256)               131328    
                                                                     
     dense_layer_3 (Dense)       (None, 128)               32896     
                                                                     
     dense_layer_4 (Dense)       (None, 64)                8256      
                                                                     
     softmax_classifier (Dense)  (None, 10)                650       
                                                                     
    =================================================================
    Total params: 697,930
    Trainable params: 697,930
    Non-trainable params: 0
    ```
""")

st.header("CNN model architecture")
st.markdown("""
    ```
    Model: "cifar10_classifier_cnn"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     convolution_1 (Conv2D)      (None, 32, 32, 32)        896       
                                                                     
     convolution_2 (Conv2D)      (None, 32, 32, 32)        9248      
                                                                     
     max_pool_1 (MaxPooling2D)   (None, 16, 16, 32)        0         
                                                                     
     batch_normalization_1 (Batc  (None, 16, 16, 32)       128       
     hNormalization)                                                 
                                                                     
     dropout_1 (Dropout)         (None, 16, 16, 32)        0         
                                                                     
     convolution_3 (Conv2D)      (None, 16, 16, 64)        18496     
                                                                     
     convolution_4 (Conv2D)      (None, 16, 16, 64)        36928     
                                                                     
     max_pool_2 (MaxPooling2D)   (None, 8, 8, 64)          0         
                                                                     
     batch_normalization_2 (Batc  (None, 8, 8, 64)         256       
     hNormalization)                                                 
                                                                     
     dropout_2 (Dropout)         (None, 8, 8, 64)          0         
                                                                     
     convolution_5 (Conv2D)      (None, 8, 8, 128)         73856     
                                                                     
     convolution_6 (Conv2D)      (None, 8, 8, 128)         147584    
                                                                     
     max_pool_3 (MaxPooling2D)   (None, 4, 4, 128)         0         
                                                                     
     batch_normalization_3 (Batc  (None, 4, 4, 128)        512       
     hNormalization)                                                 
                                                                     
     dropout_3 (Dropout)         (None, 4, 4, 128)         0         
                                                                     
     flatten (Flatten)           (None, 2048)              0         
                                                                     
     dense (Dense)               (None, 128)               262272    
                                                                     
     dense_1 (Dense)             (None, 10)                1290      
                                                                     
    =================================================================
    Total params: 551,466
    Trainable params: 551,018
    Non-trainable params: 448
    _________________________________________________________________
    ```
""")

st.header("ResNet model architecture")
st.markdown("""
    ```
    Model: "cifar10_classifier_resnet"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             multiple                  1184      
                                                                     
     residual_block (ResidualBlo  multiple                 1232      
     ck)                                                             
                                                                     
     conv2d_1 (Conv2D)           multiple                  3216      
                                                                     
     residual_block_1 (ResidualB  multiple                 4768      
     lock)                                                           
                                                                     
     conv2d_2 (Conv2D)           multiple                  4640      
                                                                     
     residual_block_2 (ResidualB  multiple                 18752     
     lock)                                                           
                                                                     
     conv2d_3 (Conv2D)           multiple                  8256      
                                                                     
     filters_change_residual_blo  multiple                 119616    
     ck (FiltersChangeResidualBl                                     
     ock)                                                            
                                                                     
     flatten_1 (Flatten)         multiple                  0         
                                                                     
     dense_3 (Dense)             multiple                  5130      
                                                                     
    =================================================================
    Total params: 166,794
    Trainable params: 166,314
    Non-trainable params: 480
    _________________________________________________________________
    ```
""")