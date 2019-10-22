# learntodrive

Implementation of Using Segmentation Masks in the ICCV 2019 Learning to Drive Challenge, Antonia Lovjer, Minsu Yeom, Benedikt Schifferer. 

In this work we predict vehicle speed and steering angle given camera image frames. We use an external pre-trained neural network for semantic segmentaion, and we augment the raw images with their segmentation masks and mirror images. We ensemble three diverse neural network models to achieve the **second best performance for both MSE angle and second best performance overall in the ICCV Learning to Drive challenge**. Our three neural network models are (i) a CNN using a single image and its segmentation mask, (ii) a stacked CNN taking as input a sequence of images and segmentation masks, and (iii) a bidirectional GRU, extracting image features using a pre-trained ResNet34, DenseNet121 and our own CNN single image model.

The models used image frames from the front-facing camera views, as in the examples shown below. The full dataset contained side images, as well as geo-location information and was provided by the ICCV competition organizers [1,2].

We subsampled the dataset on a time scale 1:10, and downsampled the images 1:12. Using a pre-trained segmentation model from NVIDIA, we generated segmentation masks for our data sample. A few examples of the model inputs are shown below. The code for the pretrained model can be found here: https://github.com/NVIDIA/semantic-segmentation

![Image Examples](https://github.com/AntoniaLovjer/learntodrive/blob/master/images/frontal_images_with_segmentation_example.png)

# Models

Model 1: CNN using a single image and its segmentation mask

![Model 1](https://github.com/AntoniaLovjer/learntodrive/blob/master/images/Network%20Architecture%201%20-%20A.png)

Model 2: Stacked CNN taking as input a sequence of images and segmentation masks

![Model 2](https://github.com/AntoniaLovjer/learntodrive/blob/master/images/Network%20Architecture%201%20-%20B.png)

Model 3: Bidirectional GRU, extracting image features using a pre-trained ResNet34, DenseNet121 and our own CNN single image model (model 1)

![Model 3](https://github.com/AntoniaLovjer/learntodrive/blob/master/images/Network%20Architecture%201%20-%20C.png)

# References

Dataset and competition: <br>

[1] Hecker, Simon, Dengxin Dai, and Luc Van Gool. “End-to-end learning of driving models with surround-view cameras and route planners.” Proceedings of the European Conference on Computer Vision (ECCV). 2018.

[2] Hecker, Simon, Dengxin Dai, and Luc Van Gool. “Learning Accurate, Comfortable and Human-like Driving.” arXiv preprint arXiv:1903.10995 (2019).

