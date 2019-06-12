import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import sys
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.widgets import Button

"""
x_points = np.arange(-5,6)
y_points = np.arange(-5,6)
print(x_points)
print(y_points)

X, Y = np.meshgrid(x_points, y_points)

Z = (X**2 + Y**2 )
print(Z)

plt.title("Circulo")
plt.xlabel("X")
plt.ylabel("Y")

plt.axis([-15,15,-15,15])
plt.contour(X,Y,Z)
plt.show()
"""
"""
x = np.arange(-2,3)
y = 2*np.square(x) + 2


X, Y = np.meshgrid(x,y)
Z = 2*np.square(X) - Y
print(Z)


print(x.size)

print(Z.shape)
print(X.shape)
print(Y.shape)

plt.xticks([-2,2])
plt.yticks(np.arange(0,20   ))
plt.title("Parabola")
plt.xlabel("X")
plt.ylabel("Y")
#plt.plot(x, y)
plt.contour(x,y,Z)
plt.show()
"""

"""
r = 5
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(phi), np.sin(theta))
y =  r * np.outer(np.sin(phi), np.sin(theta))
z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))
print(z.shape)
Z =  r * np.cos(theta)
"""

fig = plt.figure(figsize= (12,12))
ax = fig.add_subplot(1,2,1, projection ='3d')

phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)

def cube_shape():

    r = [-1,1]

    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)

    X = x * 1
    Y = y * 1
    Z = z * 1

    empty = np.array([X,Y,Z])

    return empty


def sphere_shape():

    r = 5

    x = r * np.outer(np.cos(phi), np.sin(theta))
    y =  r * np.outer(np.sin(phi), np.sin(theta))
    z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))

    empty = np.array([x,y,z])

    return empty

def cono_shape():

    r = 5
    x = r * np.outer(np.cos(phi), np.sin(theta))
    y =  r * np.outer(np.sin(phi), np.sin(theta))

    z = np.sqrt(x**2 + y**2) - 1

    empty = np.array([x,y,z])

    return empty

class Figures():

    def surface(self,x,y,z):
        ax = fig.add_subplot(1,1,1, projection ='3d')
        return ax.plot_surface(x,y,z)

    def contour(self,x,y,z):
        ax = fig.add_subplot(1,1,1, projection ='3d')
        return ax.contour(x,y,z)

    def esfera_3d(self, event):
        return self.surface(sphere_shape()[0],sphere_shape()[1], sphere_shape()[2])

    def esfera_contorno(self, event):
        return self.contour(sphere_shape()[0],sphere_shape()[1], sphere_shape()[2])

    def cone_3d(self, event):
        return self.surface(cono_shape()[0], cono_shape()[1], cono_shape()[2])

    def cone_contour(self, event):
        return self.contour(cono_shape()[0], cono_shape()[1], cono_shape()[2])

    def cube_3d(self, event):
        return self.surface(cube_shape()[0], cube_shape()[1], cube_shape()[2])

    def cube_contour(self, event):
        return self.contour(cube_shape()[0], cube_shape()[1], cube_shape()[2])


create_fig = Figures()

esfera_3D = plt.axes([0.7, 0.05, 0.1, 0.06])
esfera_contorno = plt.axes([0.81, 0.05, 0.1, 0.06])
cono_3D = plt.axes([0.59, 0.05, 0.1, 0.06])
cone_cont = plt.axes([0.48, 0.05, 0.1, 0.06])
cube_3D = plt.axes([0.37, 0.05, 0.1, 0.06])
cube_cont = plt.axes([0.26, 0.05, 0.1, 0.06])


button_sphere_3d = Button(esfera_3D, 'Esfera en 3d')
button2 = button_sphere_3d.on_clicked(create_fig.esfera_3d)

button_sphere_cont = Button(esfera_contorno, 'Esfera3dContorno')
button1 = button_sphere_cont.on_clicked(create_fig.esfera_contorno)

button_cone_3d = Button(cono_3D, 'Cono en 3d')
button3 = button_cone_3d.on_clicked(create_fig.cone_3d)

button_cone_cont = Button(cone_cont, 'Cono3dContorno')
button4 = button_cone_cont.on_clicked(create_fig.cone_contour)

button_cube_3d = Button(cube_3D, 'Cubo 3d')
button5 = button_cube_3d.on_clicked(create_fig.cube_3d)

button_cube_cont = Button(cube_cont, 'Cubo3dContorno')
button6 = button_cube_cont.on_clicked(create_fig.cube_contour)


#ax.legend()
#ax.set_xlabel('X', fontsize = 20)
#ax.set_ylabel('Y', fontsize = 20)
#ax.set_zlabel('Z', fontsize= 20)
#ax.set_title('')

plt.show()
