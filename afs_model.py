import matplotlib.pyplot as plt
import numpy as np
import os, sys

INF = float('inf')

# Scale total iterations of all models.
_SCALE_TOTAL_ITER = 0.1

class Model(object):
    def __init__(self, name, total_iter, times_per_iter):
        times_per_iter = [x / 1000000. for x in times_per_iter]
        # Constants
        self.name = name
        self.arrival_time = 0
        self.total_iter = int(total_iter * _SCALE_TOTAL_ITER)
        self.times_per_iter = times_per_iter
        self.throughputs = [1 / float(t) for t in times_per_iter]
        self.dps = [self.throughputs[0]] + [self.throughputs[i+1] - self.throughputs[i]
            for i in range(len(self.throughputs)-1)]
        self.rdps = [1/dp if dp != 0 else INF for dp in self.dps]
        self.speedups = [times_per_iter[0] / float(t) for t in times_per_iter]

        self.philly_request = self.max_gpus

        # Number of iterations remaining
        self.remain_iter = self.total_iter
        # Number of GPUs currently allocated
        self.gpus = 0
        # Current absolute time
        self.current_time = 0
        # The recent absolute time when is alloced zero GPU
        self.evicted_time = INF
        # The recent absolute time when is alloced one or more GPUs
        self.preempted_time = INF
        # How long has this model been evicted since arrival
        self.total_evicted = 0
        self.total_evicted_temp = 0
        # Abs time when the model is initially executed
        self.init_sched_time = INF
        # The nearest abs time when this model raises a re-allocation event.
        self.next_event_time = INF

        # Tiresias
        self.tiresias_tj = 0

        # Check validity
        self.validate()

    def init(self):
        self.remain_iter = self.total_iter
        self.gpus = 0
        self.current_time = 0
        self.evicted_time = INF
        self.preempted_time = INF
        self.total_evicted = 0
        self.total_evicted_temp = 0
        self.init_sched_time = INF
        self.next_event_time = INF


    def get_speed_up_dic(self):
        
        
        speed_up_dic = {0: 0}
        for i in range(len(self.speedups)):
            speed_up_dic[i+1] = self.speedups[i]
        return speed_up_dic



    @property
    def max_gpus(self):
        return len(self.speedups)

    @property
    def is_finished(self):
        return self.remain_iter <= 0

    @property
    def just_arrived(self):
        return self.current_time == self.arrival_time

    @property
    def running_for(self):
        if self.preempted_time == 0:
            return 0
        return self.current_time - self.preempted_time

    @property
    def starving_for(self):
        if self.evicted_time == 0:
            return 0
        return self.current_time - self.evicted_time

    @property
    def total_runtime(self):
        return self.current_time - self.arrival_time

    @property
    def egpus(self):
        if self.gpus == 0:
            return 0
        return self.speedups[self.gpus - 1]

    def num_gpus_for_speedup(self, ratio):
        required_speedup = self.speedups[-1] * ratio
        for i, s in enumerate(self.speedups):
            if s >= required_speedup:
                return i + 1
        assert(0)

    def submit(self, arrival_time, max_gpus=None, length=None, length_gpus=None):
        self.arrival_time = arrival_time
        self.current_time = arrival_time
        self.evicted_time = arrival_time
        if max_gpus is not None and max_gpus < self.max_gpus:
            self.times_per_iter = self.times_per_iter[:max_gpus]
            self.throughputs = self.throughputs[:max_gpus]
            self.speedups = self.speedups[:max_gpus]
            self.philly_request = max_gpus
        if length is not None:
            if length_gpus is None:
                length_gpus = self.max_gpus
            self.total_iter = int(length / self.tpi(length_gpus))
            if self.total_iter == 0:
                self.total_iter = 1
        self.remain_iter = self.total_iter
        return self

    def tpi(self, num_gpus):
        if num_gpus <= 0:
            return INF
        return self.times_per_iter[num_gpus - 1]

    def throughput(self, num_gpus):
        if num_gpus <= 0:
            return 0
        return self.throughputs[num_gpus - 1]

    def dp(self, num_gpus):
        return self.dps[num_gpus]

    def rdp(self, num_gpus):
        return self.rdps[num_gpus]

    def speedup(self, num_gpus):
        if num_gpus <= 0:
            return -INF
        return self.speedups[num_gpus - 1]

    def remain(self, num_gpus):
        if num_gpus <= 0:
            return INF
        return self.remain_iter * self.times_per_iter[num_gpus - 1]

    def finish_time(self):
        if self.gpus == 0:
            return INF
        return self.current_time + self.remain(self.gpus)

    def schedule(self, num_gpus, timeout=None):
        assert(num_gpus >= 0)
        assert(num_gpus <= self.max_gpus)
        assert(timeout is None or timeout >= 0)
        self.gpus = num_gpus
        fin_time = self.finish_time()
        # self.next_event_time = min(fin_time, self.next_event_time)
        # if timeout is not None:
        if timeout is None:
            # Default event time: when this model finishes
            self.next_event_time = fin_time
        else:
            # Set the nearest event only
            # assert(self.next_event_time == INF or self.next_event_time <= self.current_time)
            self.next_event_time = min(fin_time, self.current_time + timeout)
            # self.next_event_time = min(self.next_event_time, self.current_time + timeout)
        # print('schedule %s: %f, %f' % (self.name, fin_time, self.current_time))

    def continue_until(self, time):
        """Progress time until `time`."""
        time_diff = time - self.current_time
        if time_diff < 0:
            raise Exception('cur_time %d, time %d' % (self.current_time, time))
        if time_diff == 0:
            return
        if self.gpus == 0:
            if self.evicted_time == INF:
                self.tiresias_tj = self.current_time - self.preempted_time
                self.evicted_time = self.current_time
            self.preempted_time = INF
            self.current_time = time
            self.total_evicted = self.total_evicted_temp + time - self.evicted_time
        else:
            if self.init_sched_time == INF:
                self.init_sched_time = self.current_time
            if self.preempted_time == INF:
                self.tiresias_tj = 0
                self.total_evicted_temp += self.current_time - self.evicted_time
                self.total_evicted = self.total_evicted_temp
                self.preempted_time = self.current_time
            self.evicted_time = INF
            tpi = self.times_per_iter[self.gpus - 1]
            proced_iter = int(time_diff / float(tpi) + 1e-5)
            if self.remain_iter <= proced_iter:
                # Job finishes
                self.remain_iter = 0
                self.current_time += proced_iter * tpi
            else:
                # Still has remaining iterations
                self.remain_iter -= proced_iter
                if self.remain_iter == 1:
                    self.remain_iter = 0
                self.current_time = time
        assert(self.total_runtime >= self.total_evicted)

    def validate(self):
        """Validate if the model meets assumptions of simulation."""
        assert(self.arrival_time >= 0)
        assert(self.total_iter > 0)
        assert(len(self.speedups) > 0)
        assert(self.speedups[0] == 1.)
        if len(self.speedups) == 2:
            assert(self.speedups[0] <= self.speedups[1])
            return
        for i in range(len(self.speedups) - 2):
            # Speedups should be monotonic decreasing, and its gradient also should
            # be monotonic decreasing.
            s0 = self.speedups[i]
            s1 = self.speedups[i + 1]
            s2 = self.speedups[i + 2]
            assert(s0 <= s1)
            assert(s1 <= s2)
            # assert(s1 - s0 >= s2 - s1)

    def finish_info(self):
        return '%.1f,%s,%.1f,%.1f' % (self.current_time,
                                      self.name,
                                      self.current_time - self.arrival_time,
                                      self.total_evicted)

################################################################################

class VggnetModel256(Model):
    def __init__(self):
        super(VggnetModel256, self).__init__('VggnetModel256',
                153740040 // 256,
                [3020532, 1659208, 1263951, 868694,
                765689, 662684, 559679, 456675])

class GooglenetModel128(Model):
    def __init__(self):
        super(GooglenetModel128, self).__init__('GooglenetModel128',
                153740040 // 128,
                [346970, 187140, 146661, 106182,
                95118, 84055, 72992, 61929,
                58369, 54810, 51251, 47692,
                46375, 45058, 43741, 42424,
                42077, 41731, 41385, 41039])

class Inception4Model256(Model):
    def __init__(self):
        super(Inception4Model256, self).__init__('Inception4Model256',
                153740040 // 256,
                [5834222, 2988258, 2228434, 1468610,
                1288619, 1108629, 928639, 748649,
                689155, 629661, 570167, 510674,
                488419, 466165, 443911, 421657,
                402481, 383305, 364129, 344953,
                334708, 324464, 314220, 303976,
                296448, 288921, 281394, 273867,
                269776, 265685, 261594, 257504,
                252463, 247423, 242382, 237342,
                234883, 232425, 229966, 227508,
                221494, 215481, 209468, 203455,
                202402, 201350, 200298, 199246,
                195215, 191185, 187155, 183125])

class Resnet50Model128(Model):
    def __init__(self):
        super(Resnet50Model128, self).__init__('Resnet50Model128',
                153740040 // 128,
                [941170, 474975, 367447, 259920, 
                234511, 209103, 183695, 158287,
                147347, 136408, 125468, 114529,
                111929, 109329, 106729, 104129,
                101463, 98797, 96131, 93466,
                92738, 92011, 91283, 90556,
                89939, 89322, 88705, 88089])

class DeepspeechModel64(Model):
    def __init__(self):
        super(DeepspeechModel64, self).__init__('DeepspeechModel64',
                153740040 // 128,
                [619714, 415717, 385933, 356149,
                351970, 347791, 343612, 339434, 
                338022, 336610, 335198, 333786,
                329423, 325060, 320697, 316334,
                314514, 312695, 310875, 309056])

class AutoencoderModel51200(Model):
    def __init__(self):
        super(AutoencoderModel51200, self).__init__('AutoencoderModel51200',
                153740040 // 128,
                [17823579, 9131074, 6901069, 4671065,
                4069033, 3467002, 2864971, 2262940,
                2087242, 1911545, 1735847, 1560150,
                1447719, 1335289, 1222859, 1110429,
                1038625, 966821, 895017, 823213,
                803493, 783773, 764053, 744334,
                725709, 707085, 688461, 669837,
                646499, 623162, 599825, 576488,
                550876, 525265, 499653, 474042,
                460093, 446144, 432195, 418247,
                417658, 417069, 416481, 415891,
                415303, 412098, 408893, 405689,
                392820, 379952, 367083, 354215, 
                350944, 347673, 344402, 341132,
                337458, 333784, 330110, 326437,
                319036, 311635, 304234, 296833])

class TransformerModel4096(Model):
    # Deprecated.
    def __init__(self):
        super(TransformerModel4096, self).__init__('TransformerModel4096',
                153740040 // 128,
                [39936031, 20050013, 14982986, 9915959,
                8689551, 7463143, 6236735, 5010328,
                4588335, 4166343, 3744351, 3322359,
                3114202, 2906046, 2697890, 2489734,
                2367672, 2245611, 2123550, 2001489,
                1921372, 1841255, 1761138, 1681022,
                1622270, 1563518, 1504766, 1446014,
                1401928, 1357843, 1313757, 1269672,
                1232053, 1194434, 1156815, 1119196,
                1088336, 1057477, 1026617, 995758,
                975201, 954644, 934087, 913531,
                896795, 880060, 863324, 846589,
                825506, 804424, 783342, 762260,
                753078, 743896, 734714, 725532,
                710806, 696080, 681354, 666628,
                655069, 643511, 631953, 620395])

class TransformerModel256(Model):
    def __init__(self):
        super(TransformerModel256, self).__init__('TransformerModel256',
                153740040 // 128,
                [2776298, 1466578, 1137556, 808534,
                712710, 616886, 521062, 425236,
                410333, 395430, 380527, 365623,
                344699, 323775, 302851, 281928,
                271382, 260836, 250290, 239742,
                237244, 234746, 232248, 229748,
                228650, 227552, 226454, 225357,
                222478, 219599, 216720, 213839,
                213596, 213353, 213110, 212868,
                209979, 207090, 204201, 201310,
                199848, 198386, 196924, 195461])

class DcganModel256(Model):
    def __init__(self):
        super(DcganModel256, self).__init__('DcganModel256',
                900000000 // 256,
                [282031, 155580, 122417, 89254,
                81969, 74685, 67401, 60117,
                58979, 57841, 56703, 55566,
                53670, 51774, 49878, 47983,
                47107, 46232, 45356, 44481])

class ChatbotModel256(Model):
    def __init__(self):
        super(ChatbotModel256, self).__init__('ChatbotModel256',
                660000000 // 256,
                [114644, 80233, 73326, 66419])

class VideopredictionModel64(Model):
    def __init__(self):
        super(VideopredictionModel64, self).__init__('VideopredictionModel64',
                3200000 // 64,
                [1123488, 625106, 502540, 379974,
                358674, 337375, 316076, 294777,
                286352, 277927, 269502, 261078,
                258117, 255156, 252195, 249234,
                244498, 239762, 235026, 230291, 
                227535, 224779, 222023, 219267,
                219100, 218934, 218768, 218602])


class Models(object):
    """docstring for models"""
    def __init__(self):
        super(Models, self).__init__()
        
        self.choices = [VggnetModel256(),
        GooglenetModel128(),
        Inception4Model256(),
        Resnet50Model128(),
        DcganModel256(),
        VideopredictionModel64(),
        ChatbotModel256(),
        DeepspeechModel64(),
        TransformerModel256()]

        self.max_gpus = max(list(map(lambda m: m.max_gpus, self.choices)))

    def pick_random_model(self, max_gpus):
        

        selected_choices = list(filter(lambda m: m.max_gpus >= max_gpus, self.choices))

        if len(selected_choices) != 0:
            return np.random.choice(selected_choices)
        raise Exception(f"{max_gpus} do not map to a realistic model - currently max gpus are {self.max_gpus}")