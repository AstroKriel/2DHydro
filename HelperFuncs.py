import os
import numpy as np


## ############################################
## FOR DEBUGGING
## ############################################
def printMatrix(f_matrix):
  print(f_matrix.min(), f_matrix.max())

def saveMatrix(f_matrix, filepath_data):
  np.savetxt(f"{filepath_data}.txt", f_matrix, fmt="%2.9f", delimiter=",")


## ############################################
## FOR READING PARAMETER FILE
## ############################################
def isBoolean(str):
  return str.lower() in ["yes", "true", "t", "no", "false", "f"]

def str2bool(str):
  return str.lower() in ["yes", "true", "t"]

def isFrac(str):
  return "/" in str

def str2frac(str):
  return float(str[0]) / float(str[2])

def isFloat(str):
  return "." in str

def readParams(file_name):
  args = {}
  with open(file_name, "r") as f:
    f_lines = f.read().splitlines()
    for line in f_lines:
      line_elems = ( line.split("#")[0] ).split()
      if ("=" in line) and (len(line_elems) > 2):
        ## check if boolean
        if isBoolean(line_elems[2]):
          args[line_elems[0]] = str2bool(line_elems[2])
        ## check if string
        elif line_elems[2].isalpha():
          args[line_elems[0]] = line_elems[2]
        ## check if fraction
        elif isFrac(line_elems[2]):
          args[line_elems[0]] = str2frac(line_elems[2])
        ## check if float
        elif isFloat(line_elems[2]):
          args[line_elems[0]] = float(line_elems[2])
        ## check if integer
        elif line_elems[2].isdigit():
          args[line_elems[0]] = int(line_elems[2])
        else:
          raise Exception(f"Could not read the following line from 'params.txt':\n\t {line}")
  return args


## ############################################
## CREATE FOLDER
## ############################################
def createFolder(folder_name):
  if os.path.exists(folder_name):
    for file_name in os.listdir(f"{folder_name}/"):
      if "." in file_name:
        os.remove(f"{folder_name}/{file_name}")
    print(f"\t> Removed all files in folder '{folder_name}/' which already existed.")
  else:
    os.makedirs(folder_name)
    print(f"\t> Folder '{folder_name}/' created successfully.")



## END OF LIBRARY