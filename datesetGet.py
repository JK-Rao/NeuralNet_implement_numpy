import matplotlib.pyplot as plt
import numpy as np
import sys
sys.setrecursionlimit(1000000)

X_t = []
Y_t = []
current_lab = '0'
fig, ax = plt.subplots()


def on_mouse_click(event):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    print(event.xdata, event.ydata)
    X_t.append([event.xdata, event.ydata])
    if current_lab == '0':
        Y_t.append([1, 0, 0, 0])
    elif current_lab == '1':
        Y_t.append([0, 1, 0, 0])
    elif current_lab == '2':
        Y_t.append([0, 0, 1, 0])
    elif current_lab == '3':
        Y_t.append([0, 0, 0, 1])
    else:
        print("error in current_lab")

    for i in range(len(X_t)):
        if Y_t[i][0] == 1:
            ax.scatter(X_t[i][0], X_t[i][1], 20, 'r')
        elif Y_t[i][1] == 1:
            ax.scatter(X_t[i][0], X_t[i][1], 20, 'g')
        elif Y_t[i][2] == 1:
            ax.scatter(X_t[i][0], X_t[i][1], 20, 'y')
        elif Y_t[i][3] == 1:
            ax.scatter(X_t[i][0], X_t[i][1], 20, 'c')
    plt.show()


def on_key_press(event):
    print("Current lab is %s" % event.key)
    global current_lab
    current_lab = event.key


def get_points():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()
    X = np.array(X_t).T
    Y = np.array(Y_t).T
    return [X, Y]


[x_data, y_data] = get_points()
np.savetxt("x_data.txt", x_data)
np.savetxt("y_data.txt", y_data)
