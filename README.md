# Pixel Precipitation: Single Image Deraining with CVAE
### Machine Learning Project by Liam Jennings, Tabatha Viso, & Nikesh Walling
Worcester Polytechnic Institute, CS539, Prof. Kyumin Lee 

<table>
  <tr>
    <td>
      <img src="https://tenor.com/view/looking-out-your-window-rain-raining-gif-12687159.gif" alt="Gif" style="height: 300px;">
    </td>
    <td>
      <img src="https://c0.wallpaperflare.com/preview/890/584/45/traffic-rain-wet-windshield-droplet.jpg" alt="Image" style="height: 300px;">
    </td>
  </tr>
</table>

## Project Motivation
Gently watching raindrops roll down the window provides a calming and serene experience, but when driving, rain can impede clarity, even with reliable windshield wipers. In an increasingly automated world of autonomous vehicles, inclement weather, including raindrops, poses a significant challenge for computer vision and perception. Single-image deraining, a computer vision solution, comes into play to eliminate raindrops and streaks from images, restoring visual clarity. These models act as a filter on rainy images, enabling autonomous vehicles and other applications to function seamlessly in adverse weather conditions as if they were clear skies.

## Existing Image Filtering Approaches
Image filtering, a method that modifies pixel values, is commonly used in deraining, deblurring, and other similar image restoration applications. Some image filtering approaches include:
+ Adaptive filtering adjusts behavior based on rain characteristics, such as strength and direction, for better rain removal
+ Spatial filtering, such as Gaussian filters or bilateral filters, blurs images selectively to reduce rain impact while preserving scene details
+ Temporal filtering, used in video deraining, considers multiple consecutive frames to estimate rain streaks more effectively
+ Patch-based filtering processes images in independent patches for localized rain removal
+ Guided filtering uses a reference rain-free image to remove rain while preserving important
details

Machine learning and related techniques are used in conjunction with image filtering in order to restore images. These methods typically include deep learning (such as convolutional neural networks), math models (for describing rain properties and estimating rain streaks), support vector machines, and random forests.

## Challenges
The task of single image deraining comes with some challenges: One is dealing with the variety of rain features found in real-world situations. Raindrops can be of different shapes, sizes, densities, and orientations, making it hard to create a deraining model that works well for all types of rain patterns. Another problem is the lack of high-quality datasets containing pairs of clean and rainy images. Having enough data to train and test deraining models effectively is crucial, but gathering such datasets can be time-consuming and difficult. Many models are trained using computer-generated rain images. However, synthetic rain images may not capture all the complexities and nuances of real rain, which can affect how well the deraining models perform in actual rainy scenes.

Additionally, evaluating deraining algorithms can be challenging. The usual subjective evaluation methods, like visual inspection or human judgment, can be influenced by personal biases, as well as time-consuming. It would be helpful to have more objective and quantitative evaluation metrics that better align with human perception. By addressing these issues, deraining models can become more reliable tools for enhancing visibility and safety in challenging weather conditions, especially for applications like autonomous vehicles. 

Our project aims to tackle low image visibility caused by raindrop blurs on glass, such as a car windshield or camera lense.

Examples of different rain features on glass:
<table>
  <tr>
    <td>
      <img src="https://images.pexels.com/photos/4114968/pexels-photo-4114968.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="ex1" style="height: 220px;">
    </td>
    <td>
      <img src="https://images.pexels.com/photos/4420454/pexels-photo-4420454.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="ex2" style="height: 220px;">
    </td>
        <td>
      <img src="https://images.pexels.com/photos/5106936/pexels-photo-5106936.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="ex3" style="height: 220px;">
    </td>
            <td>
      <img src="https://images.pexels.com/photos/3054292/pexels-photo-3054292.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="ex3" style="height: 220px;">
    </td>
<td>
      <img src="https://images.pexels.com/photos/459277/pexels-photo-459277.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="ex3" style="height: 220px;">
    </td>
  </tr>
</table>

## Convolutional Variational Autoencoder (CVAE) Architecture
The proposed solution that we implemented is a Convolutional Variational Autoencoder. The architecture is as follows:
+ Preprocessing: Before feeding the data into the CVAE, a preprocessing step is applied to prepare the input images for further processing.

+ Encoder: The CVAE's encoder is responsible for mapping the input images into a latent space representation. It is comprised of:
  + Two convolutional layers to extract spatial information. The ReLU activation function is used in all layers to introduce non-linearity. Stride (2,2) is used in all layers to reduce the spatial dimensions of the feature maps (zero-padding is used).
  + One flatten layer reshapes the spatial information into a 1D vector
  + Two dense layers with ReLU activation functions are used to reduce the dimensionality of the feature maps while retaining hierarchal features
  + One sampling layer performs reparameterization

+ Reparameterization Trick: Reparameterization is used to ensure stochasticity while enabling backpropagation during training. Random noise sampled from a standard Gaussian distribution is applied to the mean and log-variance vectors obtained from the encoder. This generates a sample from the latent variable distribution, allowing the model to learn and explore the latent space effectively.

+ Decoder: The CVAE's decoder takes the sampled latent variable as input and aims to reconstruct the original image from this representation.
  + One dense layer inputs the sampled latent vector and projects it to a higher dimensional space
  + One reshape layer reshapes the 1D vector back into a 3D vector of size (30,45,256)
  + Three transposed convolutional layers with (2,2) strides and ReLU activation functions upscale feature maps back to the original image size
  + One output layer reintroduces the RGB color channels. Sigmoid activation function is used to make sure the output pixel values are in a range of (0,1), which represents the color intensity or saturation

Our Github can be found [here](https://github.com/tabathaviso/deraining-tools). The following images show a progression of our model's image reconstruction behavior through fine-tuning, such as adding convolutional layers, increasing the latent space dimension, and adjusting the number of epochs. 

<table>
  <tr>
    <td>
      <img src="https://i.ibb.co/8DrfJPt/image0.png" alt="ex1" style="height: 240px;">
    </td>
    <td>
      <img src="https://i.ibb.co/GvtWYsP/image6.png" alt="ex2" style="height: 240px;">
    </td>
    <td>
      <img src="https://i.ibb.co/GP8N0jK/image4.png" alt="ex3" style="height: 240px;">
    </td>
  </tr>
</table>


## Dataset
We used [this dataset](https://drive.google.com/drive/folders/1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K) of 861 rain-free and rain image pairs for training. This dataset was acquired from [this Github project](https://github.com/rui1996/DeRaindrop) that developed a different deraining method. Below are examples of the image pairs:

<table>
  <tr>
    <td>
      <img src="https://i.ibb.co/qrTTbnf/5-clean.png" alt="ex1" style="height: 320px;">
    </td>
    <td>
      <img src="https://i.ibb.co/qN89C41/5-rain.png" alt="ex2" style="height: 320px;">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="https://i.ibb.co/N3SPbZD/24-clean.png" alt="ex3" style="height: 320px;">
    </td>
        <td>
      <img src="https://i.ibb.co/M9rzF1k/24-rain.png" alt="ex4" style="height: 320px;">
    </td>
  </tr>
</table>


## Example of Training Result
<table>
  <tr>
    <td>
        <img src="https://i.ibb.co/PGQ0gsV/0-clean.png" alt="ex1" style="height: 240px;">
    </td>
    <td>
        <img src="https://i.ibb.co/TvzvW9V/0-rain.png" alt="ex2" style="height: 240px;">
    </td>
    <td>
        <img src="https://i.ibb.co/GP8N0jK/image4.png" alt="ex3" style="height: 240px;">
    </td>
  </tr>
</table>


## Example of Testing Result
<table>
  <tr>
    <td>
        <img src="https://i.ibb.co/M7CL596/test3.png" alt="ex1" style="height: 240px;">
    </td>
    <td>
        <img src="https://i.ibb.co/gddnZHb/test1.png" alt="ex2" style="height: 240px;">
    </td>
    <td>
        <img src="https://i.ibb.co/WKBpncP/test2.png" alt="ex3" style="height: 240px;">
    </td>
  </tr>
</table>


## Conclusions
The model performed somewhat successfully in training. It input images, identified significant hierarchal features of colored images, and reconstructed recognizable images without raindrop blurs. Unfortunately, when applying the model to testing data, the resulting images were not recognizable at all. The model needs significant further tuning and development to improve image reconstruction. Future work could include:
+ Different or ensemble approach:
  + Isolating regions distored by rain and then only applying the VAE to those regions
  + Using perceptual loss metric such as VGG loss or Generative Adversarial Metrics, which favor features that humans actually see rather than raw pixels, and should produce better results than methods that just compare raw pixel values like MSE.
+ Training the model with an improved dataset (more diverse image pairs, real rain instead of synthetic rain)
+ Data augmentation and/or denoising prior to training, using more advanced methods to split the dataset and ensure the model isn't becoming overfitted  
+ Further hyperparameter tuning 
  


## References
+ Wang, T., Yang, X., Xu, K., Chen, S., Zhang, Q., & Lau, R. “Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset”, CVPR 2019.
+ Porav, H., Bruls, T., & Newman, P. “I Can See Clearly Now: Image Restoration via De-Raining”
+ Li, S., Araujo, I. “Single Image Deraining: A Comprehensive Benchmark Analysis”, CVPR 2019.
+ Zhao, Z., Yanyan, W., Haijun, Z., Yi, Y., Shuicheng, Y., & Wang, M. "Data-Driven Single Image Deraining: A Comprehensive Review and New Perspectives," Pattern Recognition 2023.
+ [Derain Zoo](https://github.com/nnUyi/DerainZoo): additional Github collection of deraining methods and datasets
+ [Raindrops on Windshield](https://github.com/EvoCargo/RaindropsOnWindshield): dataset of synthetic rain image pairs on car windshields, specific to autonomous vehicle applications
+ [UCF Center for Research in Computer Vision](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/): dataset of images captured by Google Street View that were piped through the raindrop generator

(Please note: images that are not our own are hyperlinked to their original sources, all open-source.)
