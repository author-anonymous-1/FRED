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


class Callback(object):
    """docstring for Callback"""
    def __init__(self, plot_handle, fig, checkpoint):
        self.plot_handle = plot_handle
        self.fig = fig
        self.checkpoint = checkpoint
        self.selected_config = None
        self._addons = None
    def on_pick(self, event):


        solutions = get_non_dominated_solutions(self.checkpoint["SOLUTIONS"])
        problem = checkpoint["PROBLEM"]





        for ind in range(len(solutions)):
            self.plot_handle._facecolors[ind,:] = (1,1,1,1)
            self.plot_handle._edgecolors[ind,:] = tuple(list(BLUE) + [1])


        ind, *_ = event.ind

        # turning selected point red
        self.plot_handle._facecolors[event.ind,:] = (1,1,1,1)
        self.plot_handle._edgecolors[event.ind,:] = tuple(list(RED) + [1])


        d = self.plot_handle.get_offsets().data[event.ind][0]
        x,y=d

        closest_solution = solutions[ind]
        

        # closest_solution.variables.append(1.0)
        # print("warning not handling config delta")

        config = problem.solution_transformer(closest_solution)

        self.selected_config = config


        if self._addons:
            self._addons.remove()

        string = "<"
        for i in range(len(problem.obj_labels)):
            string += f"{problem.obj_labels[i]}: {round(d[i], 2)}, "
        string = string[:-2]
        string += ">"
        self._addons = fig.axes[0].text(x+0.1,y, string)

        # print(config)
        # print(closest_solution.objectives)


        self.fig.canvas.draw()
        plt.draw()

    def create_config(self, event):

        class_detail = self.selected_config


        '''
        file_name = "{}_{}_{}_{}.pkl".format(
            class_detail["num_classes"],
            ",".join(list(map(str,class_detail["class_thresholds"]))),
            ",".join(list(map(lambda r: str(r.numerator),class_detail["class_rates"]))),
            round(class_detail["clip_demand_factor"], 2))


        print(f"Saving config to {file_name}")
        '''

        with open("MCS_config.pkl", "wb") as fp:
            pickle.dump(class_detail, fp)



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


if __name__ == '__main__':
    fname = "experiment_6.pkl"



    savefig = False


    


    loaded_obj = []

    with open(fname, 'rb') as fp:

        while True:
            try:
                loaded_obj.append(pickle.load(fp))
            except Exception as e:
                print(f"Exception {e}")
                break


    header = loaded_obj[0]

    objectives = header["objectives"]

    print("workload: {}".format(header["workload"]))
    print([objective.get_name() for objective in objectives])


    # checkpoint?
    checkpoint = loaded_obj[-1]

    solutions = get_non_dominated_solutions(checkpoint["SOLUTIONS"])

    solutions = filter_solutions(sorted(solutions, key=lambda s: s.objectives[0]))

    checkpoint["SOLUTIONS"] = solutions
    solutions = get_non_dominated_solutions(checkpoint["SOLUTIONS"])

    # solutions = checkpoint["SOLUTIONS"]
    evaluations = checkpoint["EVALUATIONS"]
    computing_time = checkpoint["COMPUTING_TIME"]




    points = [solution.objectives[:] for solution in solutions]

    # points = sorted(points, key=lambda p: p[0])



    # fig, axs = plt.subplots(1,figsize=(8,6))
    



    # normalize jct

    data = list()
    for didex in range(len(objectives)):
        
        D = list(map(lambda p: p[didex], points))

        if "jct" in objectives[didex].get_name():
            D = normalize(D)

        data.append(D)


    fig = plt.figure(figsize=(8,6))

    plot_handle = None


    if len(objectives) == 2:
        ax = fig.add_subplot(111)
        plot_handle = ax.scatter(data[0],data[1],
                    color=["white"]*len(data[0]),
                    edgecolors=[colors[0]]*len(data[0]),
                    s=[200]*len(data[0]),
                    linewidth=1.75,
                    label="FLEX configurations", picker=True, zorder=5)
        ax.plot(data[0],data[1],
                color=BLUE,
                linewidth=1.75)


    else:
        ax = fig.add_subplot(111, projection='3d')  
        plot_handle = ax.scatter(data[0],data[1],data[2], color=[colors[0]]*len(data[0]), s=[200]*len(data[0]), label="FLEX configurations", picker=True)
        ax.set_zlabel(objectives[2].get_name())

        # tuple(list(BLUE) + [1])



    ellipse = Ellipse(xy=(2, 2.5), width=0.5, height=1, angle=135.0, 
                            edgecolor='k', fc='w', lw=2)
    ax.text(2,2.5,"Better", rotation=45.0, ha='center', va='center', size="xx-large")

    ax.add_patch(ellipse)


    draw_tri(offset_x=1.7, offset_y=2.2, ax=ax, scale=0.15)


    ax.set_title(f"{evaluations} Configurations Evaluated in {round(computing_time/60.0, 2)} Minutes")



    set_canvas(ax, x_label=objectives[0].get_name(), y_label=objectives[1].get_name())



    
    if savefig:
        plt.savefig(fname.replace(".pkl", ".png"), dpi=300)
    else:


        interact = Callback(plot_handle, fig, checkpoint)



        fig.canvas.mpl_connect('pick_event', interact.on_pick)

        ax_button = fig.add_axes([0.75,0.8,0.148,0.075])
        create_config_button = Button(ax_button, 'Create Config')
        create_config_button.on_clicked(interact.create_config)

        plt.show(block=True)
