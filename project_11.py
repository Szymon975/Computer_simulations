import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from random import random
from numba import njit
import time


############ FIRST PART #####################

@njit
def where_nb(condition, x, y):
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = x[i, j] if condition[i, j] else y[i, j]
    return result


@njit
def rand_choice_nb(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


def set_data(i):
    scenarios = {
        0: {'ny': 200, 'nx': 300, 'beta': 0.05, 'I': 1, 'R': 500, 'r': 10 ** (-6), 'delta_h': 10, 'n_drops': 4000,
            'avalanche': False, 'trails': False, 'noise': 'none'},
        1: {'ny': 40, 'nx': 60, 'beta': 0.05, 'I': 1, 'R': 40, 'r': 10 ** (-6), 'delta_h': 1, 'n_drops': 4000,
            'avalanche': False, 'trails': False, 'noise': 'none'},

        2: {'ny': 20, 'nx': 30, 'beta': 0.05, 'I': 1, 'R': 40, 'r': 10 ** (-6), 'delta_h': 1, 'n_drops': 1000,
            'avalanche': False, 'trails': False, 'noise': 'none'}

    }

    globals().update(scenarios.get(i))


set_data(1)

avalanche_arr = np.zeros((ny, nx))
slope_heights = np.zeros((ny, nx))
shifts = np.array([(0, -1), (-1, 0), (0, 1), (1, 0)])


def noise_exp(x, y):
    return r * y / 100 * np.exp(-10 ** (-6) * (x - nx / 2) ** 2)


def noise_half_sphere(x, y, R=10 ** 5):
    return r * (np.sqrt(R ** 2 - (x - nx / 2) ** 2 - (y - ny) ** 2) - np.sqrt(R ** 2 - nx ** 2 - ny ** 2))


for i, j in np.ndindex(ny, nx):
    if noise == 'none':
        slope_heights[i][j] = I * i
    if noise == 'exp':
        slope_heights[i][j] = I * i + noise_exp(j, i)
    if noise == 'half_sphere':
        slope_heights[i][j] = I * i + noise_half_sphere(j, i)


@njit
def boundary_cond(i, j, m, n):
    new_i, new_j = i + m, j + n
    if new_i == -1 or new_i == ny:
        return i, j
    if new_j == nx:
        new_j = 0
    if new_j == -1:
        new_j = nx - 1
    return new_i, new_j

@njit
def compute_boundary_cond_table(ny, nx, shifts):
    table = np.empty((ny, nx, 4, 2), dtype=np.int32)

    for i in range(ny):
        for j in range(nx):
            for p in range(4):
                delta_i, delta_j = shifts[p]
                new_i, new_j = boundary_cond(i, j, delta_i, delta_j)
                table[i, j, p] = new_i, new_j

    return table

boundary_cond_table = compute_boundary_cond_table(ny, nx, shifts)

@njit
def diff_heights_fun(slope_heights, avalanche_arr, boundary_cond_table):
    diff_heights = np.zeros((ny,nx,4))
    prob = np.zeros((ny,nx,4))
    for i in range(ny):
        for j in range(nx):
            for p in range(4):
                new_i, new_j = boundary_cond_table[i, j, p]
                avalanche = False
                while slope_heights[i][j] - slope_heights[new_i][new_j] > R:
                    slope_heights[i][j] -= 0.125*(slope_heights[i][j] - slope_heights[new_i][new_j])
                    slope_heights[new_i][new_j] += 0.125*(slope_heights[i][j] - slope_heights[new_i][new_j])
                    avalanche = True
                diff_heights[i][j][p] = slope_heights[i][j] - slope_heights[new_i][new_j]
                if avalanche:
                    avalanche_arr[new_i][new_j] = 1
               #evaluating probabilities
                a = diff_heights[i][j][p]
                if a >= 0:
                    prob[i][j][p] = np.exp(beta*a)
                else:
                    prob[i][j][p] = 0
                prob[0][j][p] = prob[1][j][p]
                prob[ny-1][j][p] = prob[ny-2][j][p]
            if np.sum(prob[i][j]) != 0:
                prob[i][j] = prob[i][j]/np.sum(prob[i][j])
            else:
                prob[i][j] = [ 1/4 for p in range(4)]

    return prob, avalanche_arr
@njit
def evolution_2(drops, prob, boundary_cond_table, hist=np.zeros((ny, nx))):
    a = 0
    drop_left_boundary = False
    while np.any(drops):
        a += 1
        hist = where_nb(drops == 1, np.ones_like(hist), hist)
        for i, j in np.ndindex(ny, nx):
            if drops[i][j] == 1:
                p = rand_choice_nb( [0,1,2,3] ,prob[i][j])
                drops[i][j] = 0
                i_new, j_new = boundary_cond_table[i,j,p]
                if (i_new == i and j_new == j):
                    drop_left_boundary = True
                    break
                elif a > ny+nx:
                    break
                else:
                    drops[i_new][j_new] = 1
                    break
    return hist, drop_left_boundary


history = np.zeros((ny, nx))
time1 = 0
for _ in range(n_drops):
    start_time = time.time()
    drops = np.zeros((ny, nx))
    drops[np.random.choice(ny), np.random.choice(nx)] = 1
    prob, avalanche_arr = diff_heights_fun(slope_heights, avalanche_arr, boundary_cond_table)
    start_time = time.time()
    current_hist, drop_left_boundary = evolution_2(drops, prob, boundary_cond_table)
    history += current_hist
    if drop_left_boundary:
        slope_heights = slope_heights - delta_h * current_hist
    if _ % 1000 == 0:
        print(f"step: {_}")

    end_time = time.time()
    time1 += end_time - start_time

# history = evolution(drops,prob, steps)

# create the color mesh plot
plt.figure(figsize=(8, 8 * ny / nx))
c = plt.imshow(slope_heights, cmap='jet', norm='linear', interpolation='bilinear', vmin=-ny, vmax=2 * ny,
               origin='lower')  # use coolwarm colormap and bilinear interpolation for smoothness
plt.colorbar(c, label='Height')  # add a colorbar on the side

trail_cmap = ListedColormap(['none', 'black'])
history = np.where(history > 0, True, False)

if trails:
    c = plt.imshow(history, cmap=trail_cmap, alpha=0.1, origin='lower')
    # c = plt.imshow(avalanche_arr, cmap=trail_cmap, alpha = 1, origin='lower')

avalanche_points = np.where(avalanche_arr == 1)

if avalanche:
    plt.scatter(avalanche_points[1], avalanche_points[0], color='black', marker='.', label='Avalanche')
    plt.legend(loc='upper right')

plt.title('Slope')
plt.xlabel('X')
plt.ylabel('Y')

#plt.savefig(f"/home/localhost/Desktop/Modeling/river_{nx}x{ny}_{n_drops}_A={avalanche}_T={trails}_N={noise}.png",
#            dpi=500)

plt.show()

######## 3D plot #####################

from mpl_toolkits.mplot3d import Axes3D

X = np.arange(0, nx)
Y = np.arange(0, ny)
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, slope_heights, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')
ax.set_title('3D Surface Plot of Slope Heights')
#plt.savefig(f"/home/localhost/Desktop/Modeling/river_{nx}x{ny}_{n_drops}_A={avalanche}_T={trails}_N={noise}_3D.png", dpi=500)

plt.show()

########################### SECOND PART #########################
import numpy as np
import random
from numba import njit


treshold = 50
@njit
def find_steepest_path(slope_heights, binary_arr, start_i, start_j):
    diff = -10**10
    indices = np.array([0,0], dtype=np.int32)
    hit_boundary = False
    for p in range(4):
        new_i,new_j = boundary_cond_table[start_i,start_j][p]
        curr_diff = slope_heights[start_i,start_j] - slope_heights[new_i,new_j]
        if curr_diff > diff or (curr_diff == diff) and binary_arr[new_i,new_j] != 1:
            diff = curr_diff
            m,n = new_i,new_j
    if not(m - start_i == 0 and n - start_j == 0) and abs(n-start_j) != nx-1:
        indices = np.array([m,n])
    else:
        hit_boundary = True
    return indices, hit_boundary
@njit
def make_rivers(slope_heights):
    r = np.ones((ny, nx))
    for i, j in np.ndindex(ny, nx):
        binary_arr = np.zeros((ny, nx))
        indices = np.array([i, j], dtype=np.int32)
        hit_boundary = False
        if j == 0 and i % 1 == 0:
            print(f"row {i}")
        a = 0
        while not hit_boundary:
            start_i, start_j = indices
            binary_arr[start_i, start_j] = 1
            indices, hit_boundary = find_steepest_path(slope_heights, binary_arr, start_i, start_j)
            m, n = indices
            if start_i != ny - 1 and abs(start_j - n) != nx - 1:
                r[start_i, start_j] += 1
                a += 1
            if a >= (nx + ny):
                break

    return r

r = make_rivers(slope_heights)
mask = np.zeros((ny,nx))
for i,j in np.ndindex(ny,nx):
    if r[i,j] >= treshold:
        mask[i,j] = 1

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['green','darkblue'])
plt.figure(figsize=(6, 5))
plt.imshow(mask, cmap=cmap, origin='lower')
plt.xlabel("x")
plt.ylabel("y")
blue_patch = mpatches.Patch(color='darkblue', label='River')
plt.legend(handles=[blue_patch])
plt.savefig(f"/home/localhost/Desktop/Modeling/river_{nx}x{ny}_{n_drops}_A={avalanche}_T={trails}_N={noise}_rivers.png", dpi=500)
plt.show()

