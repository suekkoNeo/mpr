import numpy as np 
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Working directory 
DATA_DIR = 'lung_ct'
patients = os.listdir(DATA_DIR)
#patients.sort()

def load_scan(data_dir_name):
    slices = [dicom.read_file(os.path.join(data_dir_name, s)) for s in os.listdir(data_dir_name)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness

    xy_spacing = slices[0].PixelSpacing
    z_spacing = slices[0].SliceThickness
    return slices, xy_spacing, z_spacing

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # trueのところだけ置換
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


## 必要性を感じない...　たぶんデータ量削ってるだけやと思う
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
  
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
  
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
   
    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask_320(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    ## 条件をみたすものを1, 満たさない物を0にして、さいごに+1。2値化している
    binary_image = np.array(image > -320, dtype=np.int8)+1

    # binary imageと同じサイズの配列を作り、ラベリング結果を格納
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    # 一番左上は空気もしくは肺組織以外であるから
    background_label = labels[0,0,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures 
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    # labels = measure.label(binary_image, background=0)
    # l_max = largest_label_volume(labels, bg=0)
    # if l_max is not None: # There are air pockets
    #     binary_image[labels != l_max] = 0

    return binary_image

def neighborhoodVoxel(segmented_lungs_320, xy_spacing, z_spacing):
    voxel_edge = 0.01
    x_edge = np.arange(1,501)*voxel_edge
    y_edge = np.arange(1,501)*voxel_edge
    z_edge = np.arange(1,501)*voxel_edge

    image_edge_x = np.arange(250,310)*xy_spacing[0]
    image_edge_y = np.arange(250,310)*xy_spacing[0]
    image_edge_z = np.arange(200,260)*z_spacing

    index_array_x = knn(x_edge,image_edge_x)
    index_array_y = knn(y_edge,image_edge_y)
    index_array_z = knn(z_edge,image_edge_z)

    return_array = np.empty((len(index_array_x),len(index_array_y), len(index_array_z)))
    for z in index_array_z:
        for y in index_array_y:
            for x in index_array_x:
                return_array[z,y,x] =  segmented_lungs_320[z,y,x]

    return return_array

def knn(target, search_range):
    return_array = []
    for target_point in target: 
        temp_sabun = np.array(search_range) - target_point
        return_array.append(np.argmin(abs(temp_sabun)))

    return return_array


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    ## turn 3d array into polygon data
    verts, faces, normals, values  = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

first_patient, xy_spacing, z_spacing = load_scan(DATA_DIR)
first_patient_pixels = get_pixels_hu(first_patient)

# segmented_lungs_800 = segment_lung_mask_800(pix_resampled, False)
segmented_lungs_320 = segment_lung_mask_320(first_patient_pixels)
knn_array = neighborhoodVoxel(segmented_lungs_320, xy_spacing, z_spacing)
reshape_array = np.ravel(knn_array)

reshape_array = reshape_array.astype(np.uint8)

with open('../mcx/bin/bbb.bin', 'wb') as temp_file:
    for item in reshape_array:
        temp_file.write(item)


# plot_3d(segmented_lungs_800, 0)
# plot_3d(segmented_lungs_1000, 0)