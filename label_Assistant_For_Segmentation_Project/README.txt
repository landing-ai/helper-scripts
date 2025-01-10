1. What is this?
This is a "label assistant" script for segmentation project type in Landinglens.   Similar to "label assistant" feature for Landinglens Object Detection models, you can use an already trained model in Landinglens to help generate the segmentation masks for the un-labled images. this can help accelerate your segmentation project label speed. 


2. How to use this tool
To speed up the labeling for segmentation. you can do the labeling procedure like this . 
step 1: upload 10 images into Landinglens projects, define the class and do the label for this 10 images. 
step 2: use F&E train / custom train (depends on how complex your image object is) to quickly train a model 
step 3: deploy this model into cloud end_point.
step 4: update the endpoing_ID and key value to your project in the script file in this 2 variables          self.endpoint_id = "ENTER_YOUR_OWN_ENDPOINT_HERE"   self.api_key = "ENTER_YOUR_OWN_KEY_HERE"
step 5: in the same directory, create a folder named "images", and put the unlabled images into this folder.
step 6: run this script in python environment.
step 7: after running, in the same directory, it will generate a new folder named "dataset". you can follow this instruction to complete the upload . https://support.landing.ai/docs/upload-labeled-images-seg



3. Q&A
Q: what is the difference of this tool with the "Smart Labeling" tool in Landinglens?
A: Smart labeling too l(https://support.landing.ai/docs/segmentation?highlight=smart%20labeling) in Landinglens can also help with auto-labeling.  It is using a pre-trained model to help segment . For some cases (especially when the segmentation object is a rare object in Internet iamges), it may not working well. So using the already trained model to help with the label generation can perform better in this scenario.  It is not a replacement to "smart labeling", but mainly to give user an extra option . 

 