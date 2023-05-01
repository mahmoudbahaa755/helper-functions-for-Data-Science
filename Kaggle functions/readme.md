Description: This script contains functions to download and extract data from Kaggle to your Google Drive account. It also has a function to move folders with specific class names from a source path to a destination path.

Usage:

Save this file to your desired directory.
Import the functions you need from this script to your main script.
Run the functions with the required arguments.
### Example usage:

from kaggle_downloader import add_data_from_kaggle_to_drive
 #Download and extract data from Kaggle
add_data_from_kaggle_to_drive('dataset-name', '/content/drive/MyDrive/Kaggle', '/content/drive/MyDrive/Classifcation_data', ['class-1', 'class-2'])

