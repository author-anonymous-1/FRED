import pickle
import matplotlib.pyplot as plt
from flex_tuner import *
from jmetal.util.solution import get_non_dominated_solutions
from matplotlib.widgets import Button
import seaborn as sns
from matplotlib.patches import Ellipse


def set_canvas(ax, x_label=None, y_label=None, x_lim=None, y_lim=None, y_ticks=None, x_ticks=None, legend_loc='best', legend_ncol=1, showgrid=True, legendfsize=15, showlegend=False):
    
    ax.set_facecolor(("#c8cbcf"))


    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    if showlegend:
        ax.legend(loc=legend_loc, ncol=legend_ncol, prop={'family': 'monospace', 'size': legendfsize})
    
    if x_label:
        ax.set_xlabel(x_label, fontsize=20, family="monospace")
    if y_label:
        ax.set_ylabel(y_label, fontsize=20, family="monospace")



    if x_ticks:
        ax.set_xticks(x_ticks[0])
        ax.set_xticklabels(x_ticks[1])

    if y_ticks:
        ax.set_yticks(y_ticks[0])
        ax.set_yticklabels(y_ticks[1])

    if x_lim:
        ax.set_xlim(x_lim[0],x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0],y_lim[1])


    if showgrid:
        ax.grid(alpha=.5, color="white", linestyle='-', zorder=0)

def normalize(D):
    D = [d/min(D) for d in D]
    return D[:]


colors = sns.color_palette("tab10")
BLUE = colors[0]
RED = colors[3]


def sol_dis(s1, s2):
    return np.linalg.norm(np.asarray(s1.objectives) - np.asarray(s2.objectives))

def draw_tri(offset_x, offset_y, ax, scale=1):
    tri_x = np.multiply([0,1,0], scale)
    tri_y = np.multiply([0,0,1], scale)

    tri_x = np.add(tri_x, offset_x)
    tri_y = np.add(tri_y, offset_y)


    print(tri_x)
    print(tri_y)

    ax.fill(tri_x,tri_y, color='k')

def filter_solutions(solutions):

    N = 20000

    print(f"WARNING: MANUAL DISTANCE SET AS {N}")

    filtered_solutions = [solutions[0]]
    for solution in solutions:
        s1 = filtered_solutions[-1]
        if sol_dis(s1, solution) > N:
            filtered_solutions.append(solution)    

    return filtered_solutions

def load_sim(fname):
    loaded_obj = []

    with open(fname, 'rb') as fp:
        while True:
            try:
                loaded_obj.append(pickle.load(fp))
            except EOFError as e:
                break
            except Exception as e:
                print(f"Exception {e}")
                break

    header = loaded_obj[0]
    objectives = header["objectives"]
    print("workload: {}".format(header["workload"]))
    print([objective.get_name() for objective in objectives])
