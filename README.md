# faceshape

model: contain pre-trained of face_contour and 68 points landmarks

networks: + models.py contain deep neural network such as ResNet, InceptionNet, ...
          + ViT.py contain transformers network

processingImage: + contour.py get image contour from dataset
                 + emb.py get 17 points embedding of image
                 + facemesh.py get face mesh image from dataset.


Requirement:

`python=3.7`

Install using conda:

`conda create -n faceshape python=3.7`

`conda activate faceshape`

`pip install -r requirement.txt`

Get Dataset:

`bash dataset.sh`

Process Dataset:
    - Contour image: `python countour.py`
    - FaceMesh image: `python facemesh.py`

Training:
    - Deep neural network: `python main.py -m resnet -b resnet -e 10`
    - Transformers: `python main.py -m transformers -b resnet -e 10` 

