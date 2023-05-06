import tensorflow as tf
import os
import cv2
import numpy as np
import time
from tensorflow import keras
import shutil
from random import shuffle
import zipfile
import matplotlib.pyplot as plt
import zipfile
from tensorflow.keras.applications.mobilenet import preprocess_input

'''
This code auto label new data for classifation task by spliting the data each image in it's class folder by using a pre trianied model
inputs:
    model dir
    new data loction
    loction of the old data to be merged
    classes  (optianal)

TO DO::
   - make function for object detection

'''

def load_trained_model(model_dir):
  '''
  this function take the model dir load it then from it know the input and the output shape
  Args
    model_dir => path of the model
  Return
    model, input_shape,classes in list(take len of it)

  '''

  model = tf.keras.models.load_model(model_dir)
  # return model
  # new edit
  input_shape = list(model.layers[0].input_shape[0])
  output_len = list(range(0, model.layers[-1].output.shape[1]))
  print('input_shape', input_shape)
  return model, input_shape[1:-1], output_len


def create_new_files(new_dir: str, classes: list):
  '''
  create dir and classes as floders inside the dir
  Args
    new_dir => 
    classes =>
  Return
    None
  
  '''

  !mkdir {new_dir}
  %cd {new_dir}
  classes.append('None_class')
  for i in classes:
    i = i.replace(' ', '_')
    !mkdir {i}
  %cd ..

def classifier(path,model,shape)-> int:
    ''''''
    img=cv2.imread(path)
    img=cv2.resize(img,dsize=shape)
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=preprocess_input(img_rgb)
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img,verbose=0)
    idx=np.argmax(pred)
    print('model output: ',pred)

    # values={k:i for i,k in train_batches.class_indices.items()}
    # print(values[idx])
    return idx,pred


def convert_classes_to_dect(classes: list, key='num') -> dict:
  '''
  take a list return in dict
  Args
    classes => classes of pridected model (list)
    key     => key of the dict be numbers or the classes from dict

  Return
    Dict
  '''
  d = {}
  print('before:', classes)
  classes = [i.replace(' ', '_') for i in classes]
  print('after:', classes)

  for i in range(len(classes)):
    if key == 'num':

      d[i] = classes[i]
    else:
      d[classes[i]] = i

  return d


# data_dir='/content/drive/MyDrive/Vehicle Classification /train'
def get_all_img_in_list(data_dir, classes=None):
  '''
  simply take data_dir and return all image inside it in a list
  Args


  Return

  '''
  if classes == None:
    classes = os.listdir(data_dir)
  print(classes)
  img_files = []

  for file_num in classes:
        # file_num=file_num.replace('_',' ')
        src_dir = os.path.join(data_dir, f"{file_num}")
        for f in os.listdir(src_dir):
              full_path = os.path.join(src_dir, f)
              if os.path.isfile(full_path):
                  # if 'aug_' in full_path:
                  #   continue
                  img_files.append(full_path)
        print(file_num.ljust(15), len(img_files))
  classes.append('None_class')

  return img_files, classes

# img_list,c=get_all_img_in_list('/content/drive/MyDrive/TAU Vehicle Dataset/train/train',['Bus','Truck'])


# this function decide if model prediction good or bad based on numbers
def check_model_output(x):
  ''''''
  x = list(np.round(np.array(x) * 100,2).astype(float))
  print('model percent:',x)
# accurcy more than 98 accept it
  if np.max(x)<70:
    print('baddd')
    return ''
  if np.max(x) >=92:
    return True
# predicted classes more than 4 reject it
  elif np.sum(x!=0)>=3:
    return False
  
  return True
  
  
def split_data_with_model(model_dir, data_dir, new_data_dir, classes=None):
  ''''''
  didnt_passed_img, passed_img, not_founded = 0, 0, 0

  # loading model
  model, shape, len_classes = load_trained_model(model_dir)
  # put all img in one list
  print(shape)
  img_list, classes = get_all_img_in_list(data_dir, classes)
  # test_img
  # img_list=['/content/drive/MyDrive/2.jpg',
  #           '/content/drive/MyDrive/1.jpg']
  # classes=['bus', 'half_cars', 'mallaky', 'mini_bus', 'mecropas', 'trella', 'jampo', 'None_class']
  if len(classes) != len(len_classes):
    for i in range(len(classes), len(len_classes)):
      classes.append(str(i))

  if not (os.path.exists(new_data_dir)):
    create_new_files(new_data_dir, classes)
    create_new_files(new_data_dir+'/'+'None_class', classes)

    # if type(classes) != dict:
    #     classes=convert_classes_to_dect(classes)

  print(type(classes))
  print(classes)

  print('_'*30)
  for img in img_list:
    if not (os.path.exists(img)) or img is None:
      not_founded += 1
      continue

    image_class, pred = classifier(img, model, shape)
    print('calss:', image_class)
    print('img old dir :'.ljust(13), img)
    if check_model_output(pred):
      print('passed')
      passed_img += 1
      print('img new dir :'.ljust(13), new_data_dir+'/' +
            classes[image_class]+'/'+img.split('/')[-1])
      shutil.copy(str(img), str(new_data_dir+'/' +
                  classes[image_class]+'/'+img.split('/')[-1]))
    else:
      print('didnt passed')
      didnt_passed_img += 1
      shutil.copy(str(img), str(new_data_dir+'/'+'None_class' +
                  '/'+classes[image_class]+'/'+img.split('/')[-1]))
      print('img new dir :'.ljust(13), new_data_dir+'/'+'None_class' +
            '/'+classes[image_class]+'/'+img.split('/')[-1])

    print('_'*30)
    print('\n')
  print('not passed img: '.ljust(15), didnt_passed_img)
  print('passed img: '.ljust(15), passed_img)
  print('img not founded '.ljust(15), not_founded)
  print('number of  img: '.ljust(15), len(img_list))


def move_folder(moving_path: str, source_path: str, classes: list) -> None:
    '''
    Moves folders with specific class names from the source path to the destination path.
    
    Args: 
        moving_path (str): The destination directory path.
        source_path (str): The source directory path.
        classes (list): A list of class names to be moved from the source path to the destination path.

    Return: 
        None
    '''
    print(moving_path)

    for i in classes:
        # Create the destination directory if it doesn't exist.
        if not os.path.exists(moving_path+f'/{i}'):
            l = moving_path+f'/{i}'
            !mkdir {l}
        # Move the contents of the source directory to the destination directory.
        shutil.move(os.path.join(source_path, i), os.path.join(moving_path, i))
