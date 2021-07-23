# styleCLIP-pytorch

This code mostly relies on pytorch implementation of Style Space Analysis https://github.com/xrenaa/StyleSpace-pytorch.git and styleCLIP official implementation https://github.com/orpatashnik/StyleCLIP.git
 
### Set docker environment
  ```
  bash docker.sh
  ```
### input prepare data 
  ```
  $DATASET_NAME FFHQ
  python GetCode.py --dataset_name $DATASET_NAME --code_type 'w' 
  python GetCode.py --dataset_name $DATASET_NAME --code_type 's' 
  python GetCode.py --dataset_name $DATASET_NAME --code_type 's_mean_std' 
  ```
  
### Latent.pt
  W plus space representation of an image of size 1, 18, 512
  Each convolutional layer in block with resolution 4, 8, 16, ..., 1024 consist of latent representation

### Text driven manipulation at styleCLIP.py
