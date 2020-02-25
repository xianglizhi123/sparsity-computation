import tensorflow as tf
import cv2
import numpy as np
from glob import glob
import os
from tensorflow.keras import backend as K
def resize(image,height,width):
    ratio = min(224/(height*1.0),224/(width*1.0))
    new_image = cv2.resize(image,(int(width*ratio),int(height*ratio)))
    image_bg = np.zeros((224,224,3),dtype=np.int)
    image_bg[:new_image.shape[0],:new_image.shape[1],:] = new_image
    return image_bg
def compute_sparsity(tensor, sparsity_threshold):
    size = 1
    for i in range(tensor.ndim):
        size = size*tensor.shape[i]
    tensor_array = tensor.flatten()
    #print(tensor_array.shape[0])
    zero_count = 0
    for i in range (tensor_array.shape[0]):
        if tensor_array[i] < sparsity_threshold:
            zero_count = zero_count + 1
    return zero_count * 1.0 / size
if __name__ == '__main__':
    image_path = r'/home/lizhi/TF-projects/cat-images/test.jpg'
    image_folder = r'/home/lizhi/TF-projects/sparsityOfFeatureMap/cat-images'
    image = cv2.imread(image_path)
    new_image = resize(image,224,224)
    #x = image.img_to_array(image)
    x = np.expand_dims(new_image, axis=0)
    vgg_model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    outputs = [layer.output for layer in vgg_model.layers]
    layers_name = [layer.name for layer in vgg_model.layers]
    input_files = glob(os.path.join(image_folder,'*jpg'))
    for file in input_files:
        image = cv2.imread(file)
        new_image = resize(image, 224, 224)
        # x = image.img_to_array(image)
        x = np.expand_dims(new_image, axis=0)
        for layer_name in layers_name:
            intermediate_layer_model = tf.keras.models.Model(inputs=vgg_model.input,outputs=vgg_model.get_layer(layer_name).output)
            intermediate_layer_output = intermediate_layer_model.predict(x)
            print(os.path.basename(file)+'-'+layer_name)
            print(intermediate_layer_output.shape)
            np.save(os.path.basename(file)+'-'+layer_name,intermediate_layer_output)
        #print(type(intermediate_layer_output))
            print(compute_sparsity(intermediate_layer_output, 0.001))
        print('***************************************************************************************************')


