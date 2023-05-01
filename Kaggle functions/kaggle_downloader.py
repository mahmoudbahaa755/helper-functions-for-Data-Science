import zipfile
import os
!pip install kaggle
import kaggle


def get_data_from_kaggle(data_name_from_kaggle: str, downloaded_data_dir: str, kaggle_api=None):
    '''
    this function connect to your accouct kaggle API then downloading input data to your drive account
    Args
      data_name_from_kaggle => name of data from kaggle URL
      downloaded_data_dir   => path to download kaggle data on it (note:drive path)
      kaggle_api            => if you dont have time to search in your drive account for kaggle Api
                              this function will do it for you
    Return
        None
    '''
    if kaggle_api == None:
        kaggle_api = lazy_researcher()
    !mkdir ~/.kaggle
    print('kaggle_api: ', kaggle_api)
    !cp {kaggle_api} ~/.kaggle/
    !kaggle datasets download - d {data_name_from_kaggle} - p  {downloaded_data_dir} - -force


def lazy_researcher(search_path='/content/drive/MyDrive', target_file='kaggle.json'):
    """
    Searches for a target file in all files inside a given search path and returns the path of the file if found.

    Args:
      search_path (str): The path to search for the target file (default: '/content/drive/MyDrive')
      target_file (str): The name of the file to search for (default: 'kaggle.json')

    Returns:
      str: The path of the target file if found, otherwise returns an error message.

    Raises:
      FileNotFoundError: If the target file is not found within the search path.
    """

    for dirpath, dirnames, filenames in os.walk(search_path):
        if target_file in filenames:
            file_path = os.path.join(dirpath, target_file)
            return file_path

    # If target file is not found, print an error message
    raise FileNotFoundError(
        f"Error: Unable to find the target file {target_file} within the search path {search_path}.")


def unzip(zip_file_path: str, extract_path: str, del_zip_file_after_finish=True):
    '''
      Extracts the contents of a zip file to the specified extract path.

      Args:
          zip_file_path (str): The path to the zip file to extract.
          extract_path (str): The path to extract the contents of the zip file to.
          del_zip_file_after_finish (bool, optional): Whether or not to delete the zip file after extraction. Defaults to True.
      Returns:
          None
      '''
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('/content/drive/MyDrive/Kaggle')

            if del_zip_file_after_finish:
                os.remove(zip_file_path)
                print("File deleted successfully.")

# not FINISHED


def add_data_from_kaggle_to_drive(data_name_from_kaggle, drive_data_dir, finale_moved_file, classes: list, kaggle_api=None):
  if not os.path.exists(drive_data_dir):
    !mkdir {drive_data_dir}
  get_data_from_kaggle(data_name_from_kaggle,
                       drive_data_dir, kaggle_api)  # correct
  print('get_data_from_kaggle')
  print(os.listdir(drive_data_dir))
  if len(os.listdir(drive_data_dir)) == 1:
    pass
  else:
    zip_file = drive_data_dir+'/'+os.listdir(drive_data_dir)[0]

    unzip(zip_file, drive_data_dir+'/')  # correct
    print('unzip')

  # src_file='/content/drive/MyDrive/Kaggle'
  moved_file = '/content/drive/MyDrive/Classifcation_data'
  move_folder(moved_file, finale_moved_file, classes)
  print('move_folder')
  shutil.rmtree(drive_data_dir)
  # if os.path.exist():

  #   shutil.rmtree('/content/drive/MyDrive/Classifcation_data/Kaggle')


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
