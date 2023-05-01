from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


import numpy as np
import copy
from datetime import datetime
from fractions import Fraction as frac


class Objective(object):
    """docstring for Objective"""
    def __init__(self, lmda, label):
        super(Objective, self).__init__()
        self._lmda = lmda
        self._label = label
    def __call__(self, entry):
        return self._lmda(entry)
    def get_name(self):
        return self._label


class FlexTune(FloatProblem):
    """docstring for FlexTune"""
    def __init__(self, total_gpus, app_list, event_queue, objectives):

        super(FlexTune, self).__init__()

        self._total_gpus = total_gpus
        self._app_list = app_list
        self._event_queue = event_queue

        self._service_times = list()
        for app_id in app_list:
            self._service_times.append(app_list[app_id].service)
        self._service_times.sort()

        self.objectives = objectives[:]
        self.number_of_objectives = len(objectives)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = [objective.get_name() for objective in objectives]

        # ['mean_pred','mean_jct']

                
    def get_bounds(self):
        return [self.lower_bound, self.upper_bound]
  
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        
        try:
            
            class_detail = self.solution_transformer(solution)

        except:

            
            # print(f"solution {solution} cannot be transformed")


            solution.objectives = [float('inf')] * self.number_of_objectives

            return solution        


        # print(f"solution: {solution} class_detail: {class_detail}")


        scheduler = AppMCScheduler(total_gpus=self._total_gpus,
                                    event_queue=copy.deepcopy(self._event_queue),
                                    app_list=copy.deepcopy(self._app_list),
                                    class_detail=class_detail,
                                    app_info_fn=None,
                                    estimate=True,
                                    verbosity=0)


        
        tick = datetime.now()
        scheduler.run()
        tock = datetime.now()



        objectives = self.__get_objective_value(scheduler, True)

        solution.objectives = objectives[:]

        return solution

    def __get_objective_value(self, scheduler, estimate):
        jct = list()
        estimation_error = list()
        unfairness = list()


        app_list = scheduler._app_list

        for app_id in app_list:
            app = app_list[app_id]

            if len(app.estimated_end_time):
                actual_jct = (app.end_time - app.submit_time).total_seconds()
                jct.append(actual_jct)
                
                if estimate:
                    estimated_jct = (app.estimated_end_time[0] - app.submit_time).total_seconds()
                    estimation_error.append(100.0 * abs(estimated_jct - actual_jct)/estimated_jct)
                else:
                    estimation_error.append(0.0)
                
                num_apps_seen_diff = app.num_apps_seen[0]/app.num_apps_seen[1]
                divided_cluster_size = scheduler._max_capacity/num_apps_seen_diff
                fair_jct = app.service/min(divided_cluster_size, app.initial_demand)
                unfairness.append(max(0, ((actual_jct/fair_jct) - 1.0)))
                
        # 0, 1, 2
        jct.sort()
        estimation_error.sort()
        unfairness.sort()


        # jct, unfairness, estimation error

        obj_vals = list()

        for objective in self.objectives:

            if "estimation" in objective.get_name():
                # print(f"{objective.get_name()}: {objective(estimation_error)}")
                obj_vals.append(objective(estimation_error))

            elif "jct" in objective.get_name():
                obj_vals.append(objective(jct))
                # print(f"{objective.get_name()}: {objective(jct)}")

            elif "unfairness" in objective.get_name():
                obj_vals.append(objective(unfairness))
                # print(f"{objective.get_name()}: {objective(unfairness)}")



        return obj_vals



    def get_name(self) -> str:
        raise NotImplementedError

    def solution_transformer(self, solution):        
        raise NotImplementedError

class FlexTuneWoHeuristics(FlexTune):
    """docstring for FlexTuneWoHeuristics"""
    def __init__(self, total_gpus, app_list, event_queue, objectives):
        super(FlexTuneWoHeuristics, self).__init__(total_gpus, app_list, event_queue, objectives)

        self.number_of_variables = 13
        self.number_of_constraints = 0

        # num_classes
        # T1,T2,T3,T4,T5
        # R1,R2,R3,R4,R5
        # Clip_factor
        self.lower_bound = [1.0] + [1.0]*5 + [0.0]*5 + [0.0] + [0.01]
        self.upper_bound = [5.5] + [max(self._service_times)]*5 + [1.0]*5 + [1.0] + [0.9]
        
    def get_name(self) -> str:
        return "FlexTuneWoHeuristics"



    def solution_transformer(self, solution):
        num_classes = int(solution.variables[0])
        Ts = solution.variables[1:1+num_classes]
        Rs = solution.variables[6:6+num_classes]
        clip_demand_factor = solution.variables[-2]
        delta = solution.variables[-1]
        
        thresholds = self.__eval_T(Ts)
        rates = self.__eval_R(Rs)

        class_detail = {"num_classes": len(thresholds),
                        "class_thresholds": thresholds,
                        "class_rates": rates,
                        "clip_demand_factor": clip_demand_factor,
                        "delta": delta}

        return class_detail


    def __eval_R(self, Rs):

        S = sum(Rs)
        rates = list(map(lambda r: r/sum(Rs), Rs))

        rates = list(map(lambda r: frac(round(r,3)).limit_denominator(10000), rates))
        rates = list(map(lambda r: frac(r, sum(rates)), rates))

        rates[-1] = frac(1,1) - sum(rates[:-1])

        assert(all(list(map(lambda r: r >= 0 and r <= 1.0, rates))))
        assert(np.isclose(float(sum(rates)), 1.0))
        return rates

    def __eval_T(self, Ts):

        thresholds = sorted(Ts)
        thresholds[-1] = float('inf')
        return thresholds

def comp_thresholds(job_sizes, cov_thresh=1.0):
    
    thresholds = []
    
    n=0
    mu=0
    s=0
    s2=0
    cov = 0

    class_num = 0
    cov_history =list()

    for i in range(len(job_sizes)):
        xn = job_sizes[i]
        s+=xn
        s2+=(xn*xn)
        n+=1
        mu = ((n-1.0)*mu + xn)/n
        var = (1.0/n)*(s2 + (n*mu*mu) - (mu*2.0*s))

        cov = var/(mu*mu)

        cov_history.append(cov)

        if cov > cov_thresh:
            thresholds.append(int(job_sizes[i-1]))
            n=0
            mu=0
            s=0
            s2=0
            class_num += 1


    thresholds = thresholds + [float('inf')]
    return thresholds, cov, cov_history


class FlexTuneWHeuristics(FlexTune):
    """docstring for FlexTuneWHeuristics"""
    def __init__(self, total_gpus, app_list, event_queue, objectives):

        super(FlexTuneWHeuristics, self).__init__(total_gpus, app_list, event_queue, objectives)
        self.number_of_variables = 4
        self.number_of_constraints = 0


        _, _, cov_history = comp_thresholds(self._service_times, cov_thresh=float('inf'))

        # T, R, Clip_factor
        self.lower_bound = [0.001, -3.0, 0.0, 0.01]
        self.upper_bound = [max(cov_history)+1.0, 3.0, 1.0, 0.9]
            

    def get_name(self) -> str:
        return "FlexTuneWHeuristics"


    def solution_transformer(self, solution):
        T = solution.variables[0]
        R = solution.variables[1]
        clip_demand_factor = solution.variables[2]
        delta = solution.variables[3]

        thresholds = self.__eval_T(T)
        rates = self.__eval_R(R, len(thresholds))

        class_detail = {"num_classes": len(thresholds),
                        "class_thresholds": thresholds,
                        "class_rates": rates,
                        "clip_demand_factor": clip_demand_factor,
                        "delta": delta}

        return class_detail



    def __eval_R(self, R, num_classes):


        rates = [np.exp(-1.0 * R * i) for i in range(num_classes)]
        rates = list(map(lambda r: r/sum(rates), rates))

        rates = list(map(lambda r: frac(round(r,3)).limit_denominator(10000), rates))
        rates = list(map(lambda r: frac(r, sum(rates)), rates))

        rates[-1] = frac(1,1) - sum(rates[:-1])

        assert(all(list(map(lambda r: r > 0 and r <= 1.0, rates))))
        assert(np.isclose(float(sum(rates)), 1.0))
        return rates


    def __eval_T(self, T):

        thresholds, *_ = comp_thresholds(self._service_times, cov_thresh=T)
        return thresholds