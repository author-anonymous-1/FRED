import sys
from helpers import flat_map
import copy
from common import Event, App, Job
import numpy as np
from GenericScheduler import AppGenericScheduler


class AppAFSScheduler(AppGenericScheduler):
    """docstring for AppAFSScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=False, estimate=False):
        super(AppAFSScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, estimate)
        print("Warning. compute_remaining_time makes 1job assumption")
                


    def compute_remaining_time(self, app, app_current_allocation):

        if app_current_allocation > 0:
            thrpt = app.jobs[0].thrpt(app_current_allocation)
            return app.remaining_service/thrpt
        return float('inf')

        # app_a.remaining_service/current_allocation[app_a.app_id] if current_allocation[app_a.app_id] > 0 else float('inf')

    def top_priority(self, current_allocation):
        while True:
            a_star = np.random.choice(self._active_apps)
            if a_star.demand > 0:
                break


        for app in self._active_apps:
            if app.app_id == a_star.app_id or app.demand == 0:
                continue


            app_a, app_b = a_star, app

            if current_allocation[app_a.app_id] == 0 and current_allocation[app_b.app_id] == 0:
                if app_a.remaining_service < app_b.remaining_service:
                    a_star = app_a
                else:
                    a_star = app_b
            else:

                # app_a_remaining_time = app_a.remaining_service/current_allocation[app_a.app_id] if current_allocation[app_a.app_id] > 0 else float('inf')
                # app_b_remaining_time = app_b.remaining_service/current_allocation[app_b.app_id] if current_allocation[app_b.app_id] > 0 else float('inf')

                app_a_remaining_time = self.compute_remaining_time(app_a, current_allocation[app_a.app_id])
                app_b_remaining_time = self.compute_remaining_time(app_b, current_allocation[app_b.app_id])

                if app_a_remaining_time >= app_b_remaining_time:
                    app_a, app_b = app_b, app_a

                    # throughput with current allocation
                    p_a, p_b = app_a.jobs[0].thrpt(current_allocation[app_a.app_id]), app_b.jobs[0].thrpt(current_allocation[app_b.app_id])


                    if current_allocation[app_a.app_id] < app_a.demand and current_allocation[app_b.app_id] < app_b.demand:
                        # throughput with extra GPU
                        p_a_p = app_a.jobs[0].thrpt(current_allocation[app_a.app_id]+1)
                        p_b_p = app_b.jobs[0].thrpt(current_allocation[app_b.app_id]+1)

                        if (p_b_p - p_b)/p_b_p > (p_a_p - p_a)/p_a_p:
                            a_star = app_b
                        else:
                            a_star = app_a

                    elif current_allocation[app_a.app_id] == app_a.demand:
                        a_star = app_b
                    elif current_allocation[app_b.app_id] == app_b.demand:
                        a_star = app_a
                    else:
                        print("shouldnt be here")
        return a_star



    def compute_allocation(self, event_time):
        
        
        total_demand = sum([app.demand for app in self._active_apps])

        residual = min(total_demand, self._max_capacity)

        app_id_to_allocation = {}
        
        for app in self._active_apps:
            app_id_to_allocation[app.app_id] = 0

    
        while residual > 0:

            app = self.top_priority(app_id_to_allocation)

            allocation = 1 if app_id_to_allocation[app.app_id] < app.demand else 0

            app_id_to_allocation[app.app_id] += allocation
            
            residual -= allocation


        assert(np.isclose(residual, 0)), residual


        return app_id_to_allocation