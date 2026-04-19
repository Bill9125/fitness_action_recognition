import os
os.system("python PatchTST_train.py --sport benchpress --subject_split True")
os.system("python PatchTST_train.py --sport deadlift --subject_split True")
os.system("python PatchTST_train.py --sport benchpress --subject_split False")
os.system("python PatchTST_train.py --sport deadlift --subject_split False")
