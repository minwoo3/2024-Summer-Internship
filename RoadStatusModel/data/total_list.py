import csv, sys, os, getpass
username = getpass.getuser()

nia_dir = f'/media/{username}/T7/2024-Summer-Internship/NIA2021'
cbtree_dir = f'/media/{username}/T7/2024-Summer-Internship/벚꽃'
nas_dir = f'/home/{username}/Public/GeneralCase/Raw'

nia_list = list(os.listdir(nia_dir))
print(len(nia_list))
print(os.path.basename(nia_dir))