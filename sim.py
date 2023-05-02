import os, sys, argparse
from heapq import heappush
import numpy as np
import csv
import pickle
sys.path.insert(1, 'schedulers/')

from MCSScheduler import AppMCScheduler, AppPracticalMCScheduler
from PriorityScheduler import AppPrioScheduler
from FairScheduler import AppFairScheduler, AppPracticalFairScheduler
from ThemisScheduler import AppThemisScheduler
from AFSScheduler import AppAFSScheduler

from helpers import gen_data_from_cdf

from common import App, Job, Event

from datetime import datetime, timedelta

from functools import partial
from fractions import Fraction as frac

from afs_model import Models


ADDITIONAL_SERVICE_PER_JOB = 0

def generate_SHA_jobs(app_id, num_jobs, service):

    jobs = {}

    alpha = 2
    total_stages = int(np.floor(np.log(num_jobs)/np.log(alpha))) + 1
    active_jobs = [int(np.floor(num_jobs/np.power(alpha, stage))) for stage in range(total_stages)]
    jobs_per_stage = [active_jobs[stage-1] - active_jobs[stage] for stage in range(1,total_stages)] + [1]
    fraction_per_stage = [np.power(alpha, stage) for stage in range(total_stages)]
    tau = service/np.dot(jobs_per_stage, fraction_per_stage)


    '''
    print(np.dot(jobs_per_stage, fraction_per_stage))
    print(jobs_per_stage)
    print(fraction_per_stage)
    print("=================")
    '''

    service_per_stage = np.multiply(tau, fraction_per_stage)
    jobs_per_stage = list(np.cumsum(jobs_per_stage))

    stage = 0
    for job_id in range(num_jobs):

        if job_id == jobs_per_stage[stage]:
            stage += 1

        jobs[job_id] = Job(app_id=app_id, job_id=job_id, service=service_per_stage[stage],
                            demand=1,
                            min_demand=1)

    return jobs


def generate_rect_jobs(app_id, num_jobs, service, max_gpus_per_job, min_gpus_per_job):
    jobs = {}

    for job_id in range(num_jobs):
        job = Job(app_id=app_id, job_id=job_id, service=ADDITIONAL_SERVICE_PER_JOB + (service/num_jobs),
                            demand=np.random.choice(max_gpus_per_job),
                            min_demand=np.random.choice(min_gpus_per_job))    
        

        raise NotImplementedError
        job.thrpt_dic = gen_thrpt_dic(job)


        jobs[job_id] = job

    return jobs


def generate_1_job(app_id, num_jobs, service, max_gpus_per_job, all_models):
    

    constrain_demand = False

    demand = np.sum(np.random.choice(max_gpus_per_job, num_jobs))

    jobs = {}
    job_id = 0


    model = all_models.pick_random_model(max_gpus=demand)


    # print(f"model.name: {model.name}\tunconstrained_demand: {demand}")


    '''
    if constrain_demand:

        model_labels = ["Vgg", "Google", "Inception", "Resnet",
                        "Dcgan", "Video", "Chat", "Deep", "Transformer"]
        max_demands = [8, 10, 52, 10, 8, 4, 2, 4, 8]

        for i, label in enumerate(model_labels):
            if label in model.name:
                demand = min(max_demands[i], demand)
                break
    '''

    job = Job(app_id=app_id, job_id=job_id, service=service,
                            demand=demand,
                            min_demand=0)    


    

    '''
    while True:
    	model = all_models.pick_random_model(max_gpus=demand)
    	# if model.name not in ["VideopredictionModel64", "TransformerModel256", "DcganModel256", "DeepspeechModel64", "Resnet50Model128"]:
    	if model.name in ["Inception4Model256", "VggnetModel256"]:
    		break
    '''

    job.thrpt_dic = [0.0] + model.speedups


    linear_speed_up=False

    if linear_speed_up:

        job.thrpt_dic = [float(i) for i in range(demand+1)]

    jobs[job_id] = job

    return jobs


def gen_workload_from_trace(fname, app_list, event_queue):
    # app trace should follow app_id,total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,deadline format
    # app list is a dictionary mapping from app_id to object App
    

    submit_time = datetime.now()

    all_models = Models()

    with open(fname, 'r') as fp:
        csvReader = csv.reader(fp)
        next(csvReader)
        for app_id, row in enumerate(csvReader):
            _,_,service,num_jobs,sleep_time = row

            app_id = int(app_id)
            service = float(service)
            num_jobs = int(num_jobs)
            sleep_time = float(sleep_time)

            # jobs = generate_rect_jobs(app_id, num_jobs, service, [1], [1])
            


            jobs = generate_1_job(app_id, num_jobs, service, [1], all_models)
            # jobs = generate_SHA_jobs(app_id, num_jobs, service)


            app = App(app_id=app_id, jobs=jobs, deadline=None)

            app_list[app.app_id] = app


            submit_time += timedelta(seconds=sleep_time)

            event = Event(event_id=app_id, event_time=submit_time, event_type=Event.APP_SUB, app_id=app_id)
            

            heappush(event_queue, event)


        print("%d Apps generated" % (app_id+1))




def gen_workload(cdf_app_service_times, cdf_num_jobs_per_app, cdf_max_gpus_per_job, cdf_min_gpus_per_job, load, num_gpus, num_apps, seed, app_list, event_queue):


    np.random.seed(seed)


    file_dir = os.path.dirname(os.path.abspath(__file__))



    app_service_times = gen_data_from_cdf(f"{file_dir}/cdfs/cdf-app-service-times-{cdf_app_service_times}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)


    num_jobs_per_app = gen_data_from_cdf(f"{file_dir}/cdfs/cdf-num-jobs-per-app-{cdf_num_jobs_per_app}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)


    max_gpus_per_job = gen_data_from_cdf(f"{file_dir}/cdfs/cdf-max-gpus-per-job-{cdf_max_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)


    min_gpus_per_job = gen_data_from_cdf(f"{file_dir}/cdfs/cdf-min-gpus-per-job-{cdf_min_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)


    all_models = Models()

    avg_interarrival_time = (np.mean(app_service_times))/((load)*num_gpus)


    print(f"avg_interarrival_time: {avg_interarrival_time} min")
    print(f"avg_service_time: {np.mean(app_service_times)} min")
    print(f"apps per hour: {60.0/avg_interarrival_time}")


    start_time = datetime.now()
    submit_time = datetime.now()
    sleep_time = 0

    with open("workload.csv",'w') as fp:

        fp.write("app_id,submit_time,service,num_jobs,sleep_time\n")
        

        for app_id in range(num_apps):

            num_jobs = num_jobs_per_app[app_id]
            service = max(int(float(app_service_times[app_id])/num_jobs), 1)*num_jobs

            # jobs = generate_rect_jobs(app_id, num_jobs, service, max_gpus_per_job, min_gpus_per_job)
            jobs = generate_1_job(app_id, num_jobs, service, max_gpus_per_job, all_models)

            app = App(app_id=app_id, jobs=jobs, deadline=None)

            app_list[app.app_id] = app


            event = Event(event_id=app_id, event_time=submit_time, event_type=Event.APP_SUB, app_id=app_id)
            
            heappush(event_queue, event)


            # app_id,submit_time,service,num_jobs,sleep_time
            fp.write(f"{app_id},{round((submit_time - start_time).total_seconds(),1)},{service},{num_jobs},{sleep_time}\n")

            sleep_time = int(max(0.01, np.random.exponential(avg_interarrival_time)))
            submit_time += timedelta(seconds=sleep_time)

        print("%d Apps generated" % (app_id+1))

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    
    parser.add_argument('-from_trace', help="1/0 generate workload from trace file", type=int, default=0)

    parser.add_argument('-cdf_app_service_times', help = "fname of app service times", type=str, default="afs")
    parser.add_argument('-cdf_num_jobs_per_app', help = "fname of num jobs per app", type=str, default="afs")
    parser.add_argument('-cdf_max_gpus_per_job', help = "fname of max gpus per job", type=str, default="afs")
    parser.add_argument('-cdf_min_gpus_per_job', help = "fname of min gpus per job", type=str, default="1GPU")

    parser.add_argument('-load', help = "load", type=float, default=0.8)
    parser.add_argument('-num_gpus', help='num_gpus', default=1, type=int)
    parser.add_argument('-num_apps', help="number of apps to generate", type=int, default=1)

    parser.add_argument('-scheduling_policy', help="Scheduling policy", type=str, default="FIFO")
    parser.add_argument('-logging', help="logging verbosity (0-2)", default=1, type=int)
    parser.add_argument('-estimate', help='whether to estimate ACTs 0/1', default=1, type=int, choices=[0,1])
    parser.add_argument('-output_file', default="sim_result.csv", type=str)
    parser.add_argument('-seed', type=int, default=4567)

    parser.add_argument('-MCS_config_file', default=None, type=str)

    args = parser.parse_args()


    scheduling_policy = args.scheduling_policy
    total_gpus=args.num_gpus
    estimate = args.estimate
    output_file = args.output_file



    app_list = {}
    event_queue = list()


    # total_gpus, event_queue, app_list, class_detail, app_info_fn="results.csv", suppress_print=False
    
    if scheduling_policy in ["MCS", "PMCS", "MCS_PRIO"]:



        if args.MCS_config_file == None:
            # class_detail = {"num_classes": 3, "class_thresholds": [1523, 5088, float('inf')], "class_rates": [frac(889,1000),frac(1,10),frac(11,1000)]}
            # class_detail = {"num_classes": 2, "class_thresholds": [3, float('inf')], "class_rates": [frac(950,1000),frac(50,1000)]}
            class_detail = {"num_classes": 2, "class_thresholds": [6,float('inf')], "class_rates": [frac(50,100),frac(50,100)],
                            "clip_demand_factor": 0.01, "delta": 0.01}

        else:
            with open(args.MCS_config_file, "rb") as fp:
                class_detail = pickle.load(fp)



        # class_detail = {"num_classes": 1, "class_thresholds": [float('inf')], "class_rates": [1.0]}
        # class_detail = {"num_classes": 3, "class_thresholds": [200,300,float('inf')], "class_rates": [0.85,0.1,0.05]}
        # class_detail = {"num_classes": 2, "class_thresholds": [720000,float('inf')], "class_rates": [0.75,0.25]}
        # class_detail = {"num_classes": 2, "class_thresholds": [36000.0, float('inf')], "class_rates": [0.75,0.25]}
        # class_detail = {"num_classes": 2, "class_thresholds": [500.0, float('inf')], "class_rates": [0.67,0.33]}
        # class_detail = {"num_classes": 2, "class_thresholds": [720000, float('inf')], "class_rates": [frac(88,100),frac(12,100)]}
        # class_detail = {"num_classes": 2, "class_thresholds": [3173, float('inf')], "class_rates": [frac(88,100),frac(12,100)]}
        # class_detail = {"num_classes": 2, "class_thresholds": [4000, float('inf')], "class_rates": [frac(47,50),frac(3,50)]}
        # class_detail = {"num_classes": 3, "class_thresholds": [1523, 5088, float('inf')], "class_rates": [frac(889,1000),frac(1,10),frac(11,1000)]}

        # class_detail = {"num_classes": 2, "class_thresholds": [4800, float('inf')], "class_rates": [0.92,0.08]}

        print(class_detail["num_classes"])
        print(class_detail["class_thresholds"])
        print(class_detail["class_rates"])


        if scheduling_policy == "MCS":
            scheduler = AppMCScheduler(total_gpus=total_gpus,
                                        event_queue=event_queue,
                                        app_list=app_list,
                                        class_detail=class_detail,
                                        app_info_fn=output_file,
                                        estimate=args.estimate)
        elif scheduling_policy == "PMCS":
            scheduler = AppPracticalMCScheduler(total_gpus=total_gpus,
                                        event_queue=event_queue,
                                        app_list=app_list,
                                        class_detail=class_detail,
                                        quantum=100,
                                        app_info_fn=output_file,
                                        estimate=args.estimate)


        elif scheduling_policy == "MCS_PRIO":

            def mcs_prio_func(app_id_to_end_times, a):
                return app_id_to_end_times[a.app_id]

            with open("MCS.csv") as fp:
                key, *fdata = fp.readlines()

            
            app_id_to_end_times = {}
            for entry in fdata:
                app_id,_,_,end_time,*_ = entry.split(',')
                app_id_to_end_times[int(app_id)] = float(end_time)


            scheduler = AppPrioScheduler(total_gpus=total_gpus,
                                        event_queue=event_queue,
                                        app_list=app_list,
                                        prio_func=partial(mcs_prio_func, app_id_to_end_times),
                                        app_info_fn=output_file,
                                        estimate=args.estimate)

        else:
            raise NotImplementedError

    elif scheduling_policy == "FIFO":
        scheduler = AppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.submit_time,
                                    app_info_fn=output_file,
                                    estimate=args.estimate)
    elif scheduling_policy == "SRTF":
        scheduler = AppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.remaining_service/a.demand,
                                    app_info_fn=output_file,
                                    estimate=args.estimate)
    elif scheduling_policy == "SRSF":
        scheduler = AppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.remaining_service,
                                    app_info_fn=output_file,
                                    estimate=args.estimate)
    elif scheduling_policy == "LAS":
        scheduler = AppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.service - a.remaining_service,
                                    app_info_fn=output_file,
                                    estimate=args.estimate)



    elif scheduling_policy == "FS":
        scheduler = AppFairScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=output_file,
                                    estimate=args.estimate)
    elif scheduling_policy == "PFS":
        scheduler = AppPracticalFairScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    quantum=1,
                                    app_info_fn=output_file,
                                    estimate=0)

    elif scheduling_policy == "THEMIS":
        scheduler = AppThemisScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=output_file,
                                    estimate=args.estimate)
    elif scheduling_policy == "AFS":
        scheduler = AppAFSScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=output_file,
                                    estimate=args.estimate)
    else:
        raise NotImplementedError


    if args.from_trace:
        gen_workload_from_trace("workload.csv", app_list, event_queue)
    else:
        gen_workload(args.cdf_app_service_times,
                    args.cdf_num_jobs_per_app,
                    args.cdf_max_gpus_per_job,
                    args.cdf_min_gpus_per_job,
                    args.load,
                    args.num_gpus,
                    args.num_apps,
                    args.seed,
                    app_list,
                    event_queue)
    
    

    print("Starting sim with %d Apps" % len(event_queue))

    tick = datetime.now()
    scheduler.run()
    tock = datetime.now()

    print(f"\nsim took {(tock - tick).total_seconds()} secs")

    print(f"average_num_jobs: {scheduler._avg_contention.avg}")

    


    if scheduler._num_finished_apps != len(app_list):
        print("Cluster is not big enough for largest job")
        sys.exit(1)


    print("\nSim ended.")