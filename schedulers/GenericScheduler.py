from heapq import heapify, heappop, heappush
import random
import os, sys
from helpers import flat_map

from datetime import datetime, timedelta

import copy

from common import Event, App, Job
import numpy as np

import pickle

from functools import partial

class AppGenericScheduler(object):
    """This class implements a Generic Scheduler for apps"""
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=False, estimate=True, verbosity=1):
        
        self._max_capacity = total_gpus
        self._avail_capacity = total_gpus
        
        self._active_apps = list()
        
        self._num_finished_jobs = 0
        self._num_finished_apps = 0


        self._app_id_to_allocation = None

        self._event_queue = event_queue
        self._end_event_queue = list()
        self._app_list = app_list
        
        self._estimate = estimate
        self._suppress_print = suppress_print
        self._verbosity = verbosity
        self._estimator = False

        self._gpu_util = {}
        self._stats_timeline_ticks = list()
        self._stats_timeline_gpu_util = list()
        self._stats_timeline_queue_length = list()


        self._init_time = datetime.now()
        self._last_event_time = datetime.now()
        self._app_info_fn = app_info_fn



        class EWMA(object):
            """docstring for EWMA"""
            def __init__(self, avg):
                super(EWMA, self).__init__()
                self.avg = avg
                self.weight_new_sample = 0.25
            def update(self, new_sample):
                self.avg = self.avg*(1.0 - self.weight_new_sample) + (new_sample * self.weight_new_sample)

        self._avg_contention = EWMA(0)



        if self._app_info_fn != None and not self._suppress_print:
            with open(self._app_info_fn,'w') as fp:
                fp.write("app_id,submit_time,start_time,end_time,estimated_start_time,estimated_end_time,fair_act,service,num_apps_seen_diff\n")


    def absolute_time(self, dtime):
        return (dtime - self._init_time).total_seconds()


    def compute_allocation(self):
        raise NotImplementedError

    @property
    def available_capacity(self):
        return self._available_capacity
            
        
    def update_remaining_service(self, event_time, app):
        
        if app.status == App.ACTIVE:

            for job in app.jobs.values():
                
                # print(f"app_id: {app.app_id} job.allocation: {job.allocation} job.thrpt(job.allocation): {job.thrpt(job.allocation)}")

                job.remaining_service -= job.thrpt(job.allocation) * (event_time - self._last_event_time).total_seconds()
                app.remaining_service -= job.thrpt(job.allocation) * (event_time - self._last_event_time).total_seconds()


                assert(job.remaining_service >= -1e6 ), job.remaining_service
                    
    def progress_active_apps(self, event_time):    


        for app in self._active_apps:
            self.update_remaining_service(event_time, app)


            app.num_apps_seen = (app.num_apps_seen[0]+len(self._active_apps), app.num_apps_seen[1]+1)

    def update_allocations(self, event_time):
        

        self._avg_contention.update(len(self._active_apps))
        
        self._app_id_to_allocation = self.compute_allocation(event_time)

        self._avail_capacity = self._max_capacity

        for app in self._active_apps:

            # (re)start_app, simply change rate, preempt_app

            # app.update_allocation(min(int(self._app_id_to_allocation[app.app_id]), app.demand))
            app.update_allocation(min((self._app_id_to_allocation[app.app_id]), app.demand))


            if app.status == App.SUBMITTED:
                
                if app.allocation > 0:
                    self.start_app(app, event_time)
                    app.status = App.ACTIVE
                else:
                    app.status = App.QUEUED

            elif app.status == App.ACTIVE:
                
                if app.allocation == 0:
                    app.status = App.PREEMPTED

            elif app.status == App.QUEUED:
                
                if app.allocation > 0:
                    self.start_app(app, event_time)
                    app.status = App.ACTIVE

            elif app.status == App.PREEMPTED:

                if app.allocation > 0:
                    self.start_app(app, event_time)
                    app.status = App.ACTIVE
            else:
                pass
            
            self._avail_capacity -= app.allocation

            
    def update_end_events(self, event_time):

        # assuming app.allocation is the most recent one and individual jobs have been assigned a rate

        
        '''
        for i, e in enumerate(self._end_event_queue):
            if e.event_type == Event.JOB_END:
                
                job = self._app_list[e.app_id].jobs[e.job_id]

                if np.isclose(job.allocation, 0) and not np.isclose(job.remaining_service, 0):
                    if len(job.attempts) > 0:
                        job.attempts[-1]["end_time"] = datetime.max
                elif np.isclose(job.allocation, 0) and np.isclose(job.remaining_service, 0):
                    
                    job.remaining_service = 0
                    job.attempts[-1]["end_time"] = event_time

                elif np.isclose(job.remaining_service, 0):

                    job.remaining_service = 0
                    job.attempts[-1]["end_time"] = event_time


                else:    
                    job.attempts[-1]["end_time"] = event_time + timedelta(seconds = job.remaining_service/job.allocation)



                e.event_time = job.attempts[-1]["end_time"]
            else:
                assert(False)

        '''
        '''        
        for app in self._active_apps:

            for job in app.jobs.values():

                if np.isclose(job.allocation, 0) and not np.isclose(job.remaining_service, 0):
                    if len(job.attempts) > 0:
                        job.attempts[-1]["end_time"] = datetime.max
                elif np.isclose(job.allocation, 0) and np.isclose(job.remaining_service, 0):
                    
                    job.remaining_service = 0
                    job.attempts[-1]["end_time"] = event_time

                elif np.isclose(job.remaining_service, 0):

                    job.remaining_service = 0
                    job.attempts[-1]["end_time"] = event_time

                else:    
                    job.attempts[-1]["end_time"] = event_time + timedelta(seconds = job.remaining_service/job.allocation)



        for i, e in enumerate(self._end_event_queue):
            if e.event_type == Event.JOB_END:
                e.event_time = self._app_list[e.app_id].jobs[e.job_id].attempts[-1]["end_time"]


        '''

        self._end_event_queue = list()

        for app in self._active_apps:

            for job in app.jobs.values():

                if job.status == Job.END:
                    continue

                projected_end_time = datetime.max

                if np.isclose(job.allocation, 0) and not np.isclose(job.remaining_service, 0):
                    projected_end_time = datetime.max
                else:

                    # if np.isclose(job.allocation, 0) and np.isclose(job.remaining_service, 0):
                        
                    #     job.remaining_service = 0
                    #     job.attempts[-1]["end_time"] = event_time

                    if np.isclose(job.remaining_service, 0):

                        job.remaining_service = 0
                        projected_end_time = event_time

                    else:    

                        try:
                            projected_end_time = event_time + timedelta(seconds = job.remaining_service/job.thrpt(job.allocation))
                        except Exception as e:
                            projected_end_time = datetime.max
                        

                    if len(job.attempts) > 0:
                        job.attempts[-1]["end_time"] = projected_end_time
                    else:
                        job.attempts.append({"end_time": projected_end_time})


                    event = Event(event_id=job.job_id, event_time=job.attempts[-1]["end_time"],
                                event_type=Event.JOB_END, app_id=app.app_id, job_id=job.job_id)


                    self._end_event_queue.append(event)        

        heapify(self._end_event_queue)

    def handle_app_sub_event(self, event):
        
        
        app = self._app_list[event.app_id]

        app.status = App.SUBMITTED
        app.on_app_submit(event.event_time)
        app.submit_time = event.event_time

        self._active_apps.append(app)

        app.num_apps_seen = (len(self._active_apps), 1)


    # have to look at this
    def start_app(self, app, event_time):
        
        if app.status == App.SUBMITTED or app.status == App.QUEUED:
            app.start_time = event_time
            
            for job in app.jobs.values():

                if job.allocation > 0:
                    projected_end_time = event_time + timedelta(seconds=job.remaining_service/job.thrpt(job.allocation))
                else:
                    projected_end_time = datetime.max

                event = Event(event_id=job.job_id, event_time=projected_end_time,
                            event_type=Event.JOB_END, app_id=app.app_id, job_id=job.job_id)

                
                heappush(self._end_event_queue, event)
        
        app.on_app_start(event_time)



    def log_app_info(self, app):

        if self._app_info_fn == None or app.status == App.FAILED or self._estimator:
            return

        with open(self._app_info_fn, 'a') as fp:

            submit_time = (app.submit_time - self._init_time).total_seconds()
            start_time = (app.start_time - self._init_time).total_seconds()
            end_time = (app.end_time - self._init_time).total_seconds()


            num_apps_seen_diff = app.num_apps_seen[0]/app.num_apps_seen[1]


            divided_cluster_size = self._max_capacity/num_apps_seen_diff
            fair_act = app.service/min(divided_cluster_size, app.initial_demand)

            


            if len(app.estimated_start_time) == 0:
                estimated_start_time = 0
                estimated_end_time = 0
            else:            
                estimated_start_time = (app.estimated_start_time[0] - self._init_time).total_seconds()
                estimated_end_time = (app.estimated_end_time[0] - self._init_time).total_seconds()

            

            fp.write(f"{app.app_id},{submit_time},{start_time},{end_time},{estimated_start_time},{estimated_end_time},{fair_act},{app.service},{num_apps_seen_diff}\n")


    def sim_estimate(self, app):

        temp_event_queue = self._event_queue
        temp_app_list = self._app_list

        self._event_queue = list()
        self._app_list = {}

        snap_shot = copy.deepcopy(self)

        for virtual_app in snap_shot._active_apps:
            snap_shot._app_list[virtual_app.app_id] = virtual_app

        snap_shot._estimate = False
        snap_shot._estimator = True
        snap_shot._suppress_print = True
        snap_shot._verbosity = 0

        def break_cond(v_app):
            if v_app.status == App.END:
                return True
            return False


        snap_shot.run(partial(break_cond, snap_shot._app_list[app.app_id]))
        
        app.update_estimates(snap_shot._app_list[app.app_id].start_time,
                            snap_shot._app_list[app.app_id].end_time)

        self._event_queue = temp_event_queue
        self._app_list = temp_app_list


    def handle_job_end_event(self, event):
        


        app = self._app_list[event.app_id]
        job = app.jobs[event.job_id]


        assert(np.isclose(job.remaining_service, 0.0, atol=0.01)), (self._active_apps[-1].app_id)
        

        app.on_job_end(job.job_id, event.event_time)
        

        self._num_finished_jobs += 1

        

        job_statuses = map(lambda j: j.status == Job.END, app.jobs.values())

        # all_jobs_have_finished
        if all(job_statuses):

            # stats for gpu util
            # self.log_cluster_stats(event.event_time)


            app.status = App.END
            app.end_time = event.event_time

            # remove from active_apps
            for i, a in enumerate(self._active_apps):
                if a.app_id == app.app_id:
                    self._active_apps.pop(i)
                    break
                
            self._num_finished_apps += 1


            self.log_app_info(app)
            # print(f"app {app.app_id} END at {event.event_time}")
        
        return "job_end->stage_inprog"


    def __pick_min_event(self):

        heap1 = self._event_queue
        heap2 = self._end_event_queue

        if len(heap1) == 0:
            return heappop(heap2)
        elif len(heap2) == 0:
            return heappop(heap1)
            # return heap1.popleft()
        
        heap1_event = heap1[0]
        heap2_event = heap2[0]

        # return heap1_event if heap1_event <= heap2_event else hea

        if heap1_event < heap2_event:
            return heappop(heap1)
        else:
            return heappop(heap2)


    def run(self, cond=lambda: False):

        # pkl_fname = self._app_info_fn.replace(".csv", ".pkl")

        # os.system(f"rm {pkl_fname}")

        while len(self._event_queue) > 0 or len(self._end_event_queue) > 0:

            event = self.__pick_min_event()

            self.progress_active_apps(event.event_time)            
            self._last_event_time = event.event_time


            self.report_progress(event)


            if event.event_type == Event.APP_SUB:
                self.handle_app_sub_event(event)
            elif event.event_type == Event.JOB_END:
                self.handle_job_end_event(event)


            self.update_allocations(event.event_time)

            self.update_end_events(event.event_time)

            if event.event_type == Event.APP_SUB and self._estimate and np.random.uniform() < 1.2:
                self.sim_estimate(app = self._app_list[event.app_id])
            if cond():
                break


    def util_print_progress(self, event):
        print(f"event_type: {event.event_type} event_time: {(event.event_time - self._init_time).total_seconds()}")
        for app in self._active_apps:
            print(f"app_id: {app.app_id} allocation: {app.allocation} app.demand: {app.demand} app.remaining_service: {app.remaining_service}")
            for job in app.jobs.values():
                print(f"\tjob_id: {job.job_id} allocation: {job.allocation} status: {job.status} job.remaining_service: {job.remaining_service}")
        print("==============================")
        


    def util_pickle_progress(self, event):

        log_fname = self._app_info_fn.replace(".csv", ".pkl")

        with open(log_fname, 'ab') as fp:

            pickle.dump([event, self._active_apps, self._init_time], fp)

     
    def report_progress(self, event):
        if self._verbosity == 1:
            print("\rlen(end_event_queue): %d \t active_apps: %d \t Apps done: %d" % (len(self._end_event_queue), len(self._active_apps), self._num_finished_apps),end='')
        elif self._verbosity == 2:
            self.util_print_progress(event)
        elif self._verbosity == 3:
            self.util_pickle_progress(event)
        elif self._verbosity == 4:
            self.util_print_progress(event)
            self.util_pickle_progress(event)                


    @property
    def total_gpus(self):
        return self._avail_capacity

    @property
    def max_gpus(self):
        return self._max_capacity
    

    @property
    def num_finished_jobs(self):
        return self._num_finished_jobs    

    @property
    def num_finished_apps(self):
        return self._num_finished_apps
