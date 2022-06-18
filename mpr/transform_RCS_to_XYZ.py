import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import optuna


data_dir_name = 'lung_ct'
files = os.listdir(data_dir_name)
slices = [pydicom.dcmread(os.path.join(data_dir_name, file)) for file in files]

slice_projections = []
for slice in slices:
    IOP = np.array(slice.ImageOrientationPatient)
    IPP = np.array(slice.ImagePositionPatient)
    normal = np.cross(IOP[0:3], IOP[3:]) # IOPに従って方向づけされた際のx,y軸に対して、垂直なz軸を決定する
    projection = np.dot(IPP, normal) # z軸との平面の距離を決定
    slice_projections += [{"d": projection, "slice": slice}]

sorted_slices = sorted(slice_projections, key=lambda i: i["d"])
sorted_instance = np.dstack([slice["slice"].pixel_array for slice in sorted_slices])

plot_slices(sorted_instance)