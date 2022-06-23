import numpy as np 
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

temp = []
with open('../bessel_test.mc2', 'rb') as f:
    data=np.fromfile(f,dtype=np.uint32)
    array=np.reshape(data*100, (500,500,500))
    for z_array in array:
        for y_array in z_array: 
            for x_array in y_array:
                if x_array != 0:
                    print(x_array)


def plot_3d(image, threshold=0):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    ## turn 3d array into polygon data
    # verts, faces, normals, values  = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10000, 10000))
    ax = fig.add_subplot(111, projection='3d')

    # # Fancy indexing: `verts[faces]` to generate a collection of triangles
    # mesh = Poly3DCollection(verts[faces], alpha=0.70)
    # face_color = [0.45, 0.45, 0.75]
    # mesh.set_facecolor(face_color)
    # ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

plot_3d(array)