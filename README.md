# deepfake_faceswap

Using GAN Network Trained a model to swap faces.

# Download dataset # 

To Download the dataset  [https://anonfile.com/p7w3m0d5be/face-swap.zip]

# Run the Network #

#### Extraction ####
First we need to extract faces from the input images. To do so run the following command

` python faceswap.py extract <input_model_directory> `

#### Train the Network ####
You will have two folders in input model directory. Run twice to extract faces from the images. **python faceswap.py extract /path_to_the_file/folder_1** and do same thing for **folder_2**.

For **extract.py** file you will need few libraries. Check all libraries you want to run this code. One important library is **face_recognition** library which is pre-written and you can use just by `pip install face_recognition`. Any Trouble installing the library follow this link [https://github.com/ageitgey/face_recognition]

Now Face will extracted from the input image and store in **output** folder. We can feed those image to our Network. To train two folder image dataset run the following command

`python faceswap.py train --input-A <path_to_folder_A> --input-B <path_to_folder_B> --model-dir <path_to_store_models> -bs 64`

Now the network will start Training and save the weights in the model directory. Before running this file make sure you have installed all dependencies for Opencv, Tensorflow and Keras. The model is purely written on Tensorflow library. Make sure all libraries and plugins are available in your local directory.

#### Convert Original Image to Swap the faces ####
To Swap face run the following command

`python faceswap.py convert --model-dir <path_to_stored_model> --converter GAN --detector hog --frame-ranges 10-50`

If you followed every step then by running simply `python faceswap.py convert` will do the job. When you customized the code you should mention in the command prompt for better understanding.

### NOTE ### 
---
To run this model you need heavy GPU computing power. The model should run atleast > 100000 epochs to get fair model. Try Google cloud or other GPU platform which you are familiar with. The model will train atleast >50hrs. And to generate a swap in better quality video it will take 1hour to complete **1 min** video content. Dont try to run this on local System unless you have **GPU** yourself.
