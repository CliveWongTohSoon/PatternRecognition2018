import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot(X, Y, x_dim0=0, x_dim1=1, color_map='nipy_spectral'):
    x_min, x_max = X[:, x_dim0].min() - .5, X[:, x_dim0].max() + .5
    y_min, y_max = X[:, x_dim1].min() - .5, X[:, x_dim1].max() + .5
    plt.figure(figsize=(8, 6))

    # clean the figure

    plt.scatter(X[:, x_dim0], X[:, x_dim1], c=Y, cmap=color_map)
    plt.xlabel(f'Feature {x_dim0}')
    plt.ylabel(f'Feature {x_dim1}')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

def plot_3d(X, Y, x_dim=(0,1,2), color_map='nipy_spectral', subplot=111):
    x_min, x_max = X[:, x_dim[0]].min() - .5, X[:, x_dim[0]].max() + .5
    y_min, y_max = X[:, x_dim[1]].min() - .5, X[:, x_dim[1]].max() + .5
    z_min, z_max = X[:, x_dim[2]].min() - .5, X[:, x_dim[2]].max() + .5
    
    fig = plt.figure(figsize=(8, 6))

    # clean the figure
#     plt.clf()
    
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[:, x_dim[0]], X[:, x_dim[1]], X[:, x_dim[2]], c=Y, cmap=color_map)
    ax.set_xlabel(f'Feature {x_dim[0]}')
    ax.set_ylabel(f'Feature {x_dim[1]}')
    ax.set_zlabel(f'Feature {x_dim[2]}')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    
    plt.show()