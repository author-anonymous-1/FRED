import sys
import copy
from common import Event, App, Job
import numpy as np
from heapq import heapify, heappop, heappush
from GenericScheduler import AppGenericScheduler
from fractions import Fraction as frac
from datetime import datetime, timedelta


# self, total_gpus, event_queue, app_list, suppress_print=False


class AppMCScheduler(AppGenericScheduler):
    """This class implements a Multi-Class Scheduler with fixed rate for Apps"""
   
    def __init__(self, total_gpus, event_queue, app_list, class_detail, app_info_fn="results.csv", suppress_print=False, estimate=False, verbosity=1):
        super(AppMCScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, estimate, verbosity=verbosity)

        self._class_detail = copy.deepcopy(class_detail)
        self._num_classes = class_detail["num_classes"]
        self._class_thresholds = class_detail["class_thresholds"]
        self._default_class_rates = copy.copy(class_detail["class_rates"])
        self._clip_demand_factor = class_detail["clip_demand_factor"]
        self._delta = class_detail["delta"]

        assert(np.isclose(float(sum(self._default_class_rates)), 1.0)), float(sum(self._default_class_rates))
        
        for i in range(self._num_classes):
            self._default_class_rates[i] = self._default_class_rates[i] * self._max_capacity
        
        self._class_rates = self._default_class_rates[:]

    def get_virtual_rates(self):

        residual_rate = 0
        
        virtual_rates = [0]*self._num_classes

        for i in range(self._num_classes):
            if len(self._active_apps[i]) > 0:
                virtual_rates[i] = self._default_class_rates[i]
            else:
                residual_rate += self._default_class_rates[i]

        return list(map(lambda vr: vr * (1.0/(1.0 - residual_rate)), virtual_rates))


    def analytical_estimate(self, app, event_time):

        # virtual_rates
        vr = self.get_virtual_rates()
        S = [0]*len(vr)
        for i in range(self._num_classes):
            for active_app in self._active_apps[i]:
                S[i] += active_app.remaining_service

        k = app.app_class

        start_time = 0
        end_time = 0
        for i in range(self._num_classes):
            if vr[i] != 0:
                start_time += vr[i] * min(S[k]/vr[k], S[i]/vr[i])

                if i == k:
                    end_time += S[k]
                else:
                    end_time += vr[i] * min((S[k])/vr[k], S[i]/vr[i])                    
            else:
                assert(S[i] == 0), S[i]


        # start_time
        app.estimated_start_time = event_time + start_time
        app.estimated_end_time = event_time + end_time
        

    def __average_service(self, app):
        app_service = app.service
        

        s = 0
        for i in range(1,len(app.jobs[0].thrpt_dic)):
            s += i/app.jobs[0].thrpt_dic[i]

        return (app_service/(i-1)) * s

    # or app_service = class_rate * app.service/thrpt(class_rate)


    def __clip_demand(self, app, clip_demand_factor=0.9):
        

        for job in app.jobs.values():
            job.optimal_demand = job.demand
            for g in range(1,job.demand+1):
                if job.thrpt_dic[g]/float(g) < clip_demand_factor:
                    job.optimal_demand = g-1
                    break
        
        app.optimal_demand = sum([job.optimal_demand if job.status != Job.END else 0 for job in app.jobs.values()])

    def classifier(self, app, induce_error=False):

        app_service = app.service

        # app_service = self.__average_service(app)

        p_error = 40.0
        
        if induce_error:
            app_service *= (1.0 + np.random.uniform(-1.0*(p_error/100.0),(p_error/100.0)))

            assert(app_service>0)

        for i, t in enumerate(self._class_thresholds):
            if app_service <= t:
                return i
        return i

    def handle_app_sub_event(self, event):
        super(AppMCScheduler, self).handle_app_sub_event(event)
        app = self._app_list[event.app_id]
        app.app_class = self.classifier(app)
        # self.__clip_demand(app)
        
    def compute_allocation_non_work_serving(self, event_time):

        class_demand = [0] * self._num_classes                
        class_allocation = [0] * self._num_classes
        self._class_rates = self._default_class_rates[:]

        for app in self._active_apps:
            class_demand[app.app_class] += app.demand


        total_demand = sum(class_demand)
        residual = sum(self._default_class_rates)


        tries = 0

        while residual > 0 and total_demand > 0:
            for i in range(self._num_classes):

                allocation = min(self._class_rates[i], class_demand[i])

                if np.isclose(float(allocation), 0):
                    allocation = 0

                class_allocation[i] += allocation
                class_demand[i] -= allocation
                residual -= allocation
                total_demand -= allocation

                if np.isclose(float(class_demand[i]), 0):
                    self._class_rates[i] = 0
                    class_demand[i] = 0


            if np.isclose(float(residual), 0.0) or np.isclose(float(total_demand), 0.0):
                break


            R = sum(self._class_rates)
            if R > 0:
                self._class_rates = [frac(residual * self._class_rates[i], R) for i in range(self._num_classes)]

            tries += 1

            if tries > 100:
                break
                # raise Exception("Too many while loops")






        self._class_rates = class_allocation[:]
        app_id_to_allocation = {}
        
        for app in self._active_apps:
            # app_id_to_allocation[app.app_id] = min(class_allocation[app.app_class] if class_allocation[app.app_class] >= app.min_demand else 0, app.demand)
            app_id_to_allocation[app.app_id] = float(min(class_allocation[app.app_class], app.demand))
            class_allocation[app.app_class] -= app_id_to_allocation[app.app_id]

        assert(np.isclose(float(sum(class_allocation)), 0)), class_allocation


        return app_id_to_allocation



    def __intra_class_allocation_afs_style(self, app_class, class_allocation, app_id_to_allocation):
        class_apps = list(filter(lambda a: a.app_class == app_class, self._active_apps))
        
        # allocate guaranteed
        for app in class_apps:
            app_id_to_allocation[app.app_id] = float(min(class_allocation, app.optimal_demand))
            class_allocation -= app_id_to_allocation[app.app_id]

        # allocate leftover
        while class_allocation > 0.0:
            
            potential_allocation = min(1.0, class_allocation)

            class_apps = sorted(class_apps, key = lambda a: (a.jobs[0].thrpt(app_id_to_allocation[a.app_id] + potential_allocation) - a.jobs[0].thrpt(app_id_to_allocation[a.app_id]), -1*a.app_id) )
            
            optimal_app = class_apps[-1]
            app_id_to_allocation[optimal_app.app_id] += potential_allocation

            assert(optimal_app.demand >= app_id_to_allocation[optimal_app.app_id])
            
            class_allocation -= potential_allocation

        assert(np.isclose(float((class_allocation)), 0.0)), class_allocation


    def __intra_class_allocation(self, app_class, class_allocation, app_id_to_allocation):
        class_apps = list(filter(lambda a: a.app_class == app_class, self._active_apps))
        

        # sort by who came first
        class_apps = sorted(class_apps, key=lambda a: a.app_id)


        delta=self._delta
        clip_demand_factor = self._clip_demand_factor

        while class_allocation > 0.0:

            for app in class_apps:
                self.__clip_demand(app, clip_demand_factor)
                app_id_to_allocation[app.app_id] = float(min(class_allocation, app.optimal_demand))
                class_allocation -= app_id_to_allocation[app.app_id]

            clip_demand_factor -= delta


        for app in class_apps:
            assert(app.demand >= app_id_to_allocation[app.app_id]) 
        assert(np.isclose(float((class_allocation)), 0.0)), class_allocation

    def compute_allocation(self, event_time):

        class_demand = [0] * self._num_classes                
        class_allocation = [0] * self._num_classes
        self._class_rates = self._default_class_rates[:]

        for app in self._active_apps:
            class_demand[app.app_class] += app.demand


        total_demand = sum(class_demand)
        residual = sum(self._default_class_rates)


        tries = 0

        while residual > 0 and total_demand > 0:
            for i in range(self._num_classes):

                allocation = min(self._class_rates[i], class_demand[i])

                if np.isclose(float(allocation), 0):
                    allocation = 0

                class_allocation[i] += allocation
                class_demand[i] -= allocation
                residual -= allocation
                total_demand -= allocation

                if np.isclose(float(class_demand[i]), 0):
                    self._class_rates[i] = 0
                    class_demand[i] = 0


            if np.isclose(float(residual), 0.0) or np.isclose(float(total_demand), 0.0):
                break


            R = sum(self._class_rates)
            if R > 0:
                self._class_rates = [frac(residual * self._class_rates[i], R) for i in range(self._num_classes)]

            tries += 1

            if tries > 100:
                break
                # raise Exception("Too many while loops")

        # after this while loop, we have gpu allocations per class in the class_allocation vector


        self._class_rates = class_allocation[:]
        app_id_to_allocation = {}

        for app_class in range(self._num_classes):
            self.__intra_class_allocation(app_class, class_allocation[app_class], app_id_to_allocation)

        return app_id_to_allocation

class AppPracticalMCScheduler(AppGenericScheduler):
    """docstring for AppPracticalMCScheduler"""

    def __init__(self, total_gpus, event_queue, app_list, class_detail, quantum, app_info_fn="results.csv", suppress_print=False, estimate=False):
        super(AppPracticalMCScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, estimate)

        self._class_detail = copy.deepcopy(class_detail)
        self._num_classes = class_detail["num_classes"]
        self._class_thresholds = class_detail["class_thresholds"]
        self._default_class_rates = copy.copy(class_detail["class_rates"])
        assert(np.isclose(float(sum(self._default_class_rates)), 1.0)), float(sum(self._default_class_rates))
        
        for i in range(self._num_classes):
            self._default_class_rates[i] = self._default_class_rates[i] * self._max_capacity
        
        self._class_rates = self._default_class_rates[:]


        self._redivision_event_queue = list()
        self._quantum = quantum
        self._app_id_to_fair_allocation = {}
        self._app_id_to_fractional_allocation = {}
        self._app_id_to_int_allocation = {}
        self._fractional_share = None
        self._sharing_group = list()

    def classifier(self, app):
        for i, t in enumerate(self._class_thresholds):
            if app.service <= t:
                return i
        return i

    def handle_app_sub_event(self, event):
        super(AppPracticalMCScheduler, self).handle_app_sub_event(event)
        app.app_class = self.classifier(app)

    def compute_MCS_allocation(self, event_time):

        class_demand = [0] * self._num_classes                
        class_allocation = [0] * self._num_classes
        self._class_rates = self._default_class_rates[:]

        for app in self._active_apps:
            class_demand[app.app_class] += app.demand


        total_demand = sum(class_demand)
        residual = sum(self._default_class_rates)

        while residual > 0 and total_demand > 0:
            for i in range(self._num_classes):

                allocation = min(self._class_rates[i], class_demand[i])

                class_allocation[i] += allocation
                class_demand[i] -= allocation
                residual -= allocation
                total_demand -= allocation

                if class_demand[i] == 0:
                    self._class_rates[i] = 0


            if np.isclose(float(residual), 0.0) or np.isclose(float(total_demand), 0.0):
                break


            R = sum(self._class_rates)
            if R > 0:
                self._class_rates = [frac(residual * self._class_rates[i], R) for i in range(self._num_classes)]



        self._class_rates = class_allocation[:]
        app_id_to_allocation = {}
        
        for app in self._active_apps:
            # app_id_to_allocation[app.app_id] = min(class_allocation[app.app_class] if class_allocation[app.app_class] >= app.min_demand else 0, app.demand)
            app_id_to_allocation[app.app_id] = min(class_allocation[app.app_class], app.demand)
            class_allocation[app.app_class] -= app_id_to_allocation[app.app_id]

        assert(np.isclose(float(sum(class_allocation)), 0)), float(class_allocation)


        return app_id_to_allocation


    def compute_allocation(self, event_time):


        app_id_to_mcs_allocation = self.compute_MCS_allocation(event_time)

        residual = self._fractional_share
        app_id_to_allocation = {}


        # assign int allocation
        for app_id in self._app_id_to_int_allocation:
            app_id_to_allocation[app_id] = int(self._app_id_to_int_allocation[app_id])


        # these apps are those with fractional share
        total_remaining_demand = sum([app.demand - app_id_to_allocation[app.app_id] for app in self._sharing_group])

        for app in self._sharing_group:

            remaining_demand = app.demand - app_id_to_allocation[app.app_id]
            additional_allocation = min(residual, remaining_demand, 1)

            app_id_to_allocation[app.app_id] += additional_allocation
            residual -= additional_allocation
            total_remaining_demand -= additional_allocation

        return app_id_to_allocation


    def __pick_min_heap(self, heap1, heap2):

        if len(heap1) == 0:
            return heap2
        elif len(heap2) == 0:
            return heap1
        
        heap1_event = heap1[0]
        heap2_event = heap2[0]

        if heap1_event < heap2_event:
            return heap1
        else:
            return heap2


    def redivision(self, event):
        total_allocation = self._fractional_share

        # left shift
        if len(self._sharing_group) > 1:
            
            self._sharing_group.append(self._sharing_group.pop(0))

            if not np.isclose(float(total_allocation), 0.0):

                next_app = self._sharing_group[0]

                
                next_redivision = (self._quantum * float(self._app_id_to_fractional_allocation[next_app.app_id]))

                assert(float(next_redivision) >= 0)

                # for app in self._active_apps:
                #     print(f"app_id: {app.app_id} fair_share: {self._app_id_to_fair_allocation[app.app_id]} frac_share: {self._app_id_to_fractional_allocation[app.app_id]}")

                self._redivision_event_queue = list()
                # heappush(self._redivision_event_queue, Event(event_id=0, event_time=event.event_time + timedelta(seconds=float(next_redivision)), event_type="REDIVISION"))
                heappush(self._redivision_event_queue, Event(event_id=0, event_time=event.event_time + timedelta(seconds=float(next_redivision)), event_type="REDIVISION"))




    def run(self, cond=lambda: False):

        while len(self._event_queue) > 0 or len(self._end_event_queue) > 0:

            event = heappop(self.__pick_min_heap(self.__pick_min_heap(self._event_queue, self._end_event_queue), self._redivision_event_queue))

            self.progress_active_apps(event.event_time)            
            self._last_event_time = event.event_time

            self.report_progress(event)

            if event.event_type == Event.APP_SUB:
                self.handle_app_sub_event(event)

            elif event.event_type == Event.JOB_END:
                self.handle_job_end_event(event)



            if event.event_type in [Event.APP_SUB, Event.JOB_END, "REDIVISION"]:

                if event.event_type in [Event.APP_SUB, Event.JOB_END]:
                    

                    self._app_id_to_MCS_allocation = {}
                    self._app_id_to_fractional_allocation = {}
                    self._app_id_to_int_allocation = {}
                    self._fractional_share = None
                    self._sharing_group = list()

                    self._app_id_to_MCS_allocation = self.compute_MCS_allocation(event.event_time)
                    

                    for app_id in self._app_id_to_MCS_allocation:
                        self._app_id_to_fractional_allocation[app_id] = self._app_id_to_MCS_allocation[app_id] - int(self._app_id_to_MCS_allocation[app_id])

                        self._app_id_to_int_allocation[app_id] = int(self._app_id_to_MCS_allocation[app_id])

                        if self._app_id_to_fractional_allocation[app_id] > 0:
                            self._sharing_group.append(self._app_list[app_id])

                    self._fractional_share = int(sum([self._app_id_to_fractional_allocation[app_id] for app_id in self._app_id_to_fractional_allocation]))

                self.redivision(event)

            self.update_allocations(event.event_time)

            self.update_end_events(event.event_time)

            if event.event_type == Event.APP_SUB and self._estimate and np.random.uniform() < 1.2:
                self.sim_estimate(app = self._app_list[event.app_id])

            if cond():
                break