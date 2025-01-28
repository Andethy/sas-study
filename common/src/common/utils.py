import numpy as np
import math
import time
from pythonosc import udp_client

from common.osc import OSCManager

# Timing Mods #
global stepT
stepT = 0.004


class robotsUtils:
    def __init__(self, IPadress, *args, sim=False):
        self.arm = IPadress
        self.initPos = [0, 0, 0, 90, 0, 0, 0]
        if sim == False:
            from xarm.wrapper import XArmAPI
            self.xArm = XArmAPI(self.arm)
        self.sim = sim
        self.IPtoSEND = "127.0.0.1"  # "192.168.1.50"
        per = 4
        self.ports = tuple(10001 + i for i in range(per)) + tuple(11001 + i for i in range(per)) + tuple(
            12001 + i for i in range(per)) + tuple(1001 + i for i in range(per))
        # this is indexed at 1
        self.routes = (('melody',) * 8) + (('pitch',) * 8)
        self.client = ...
        self.clients = OSCManager(self.IPtoSEND, *args)

    def setupBot(self):
        self.xArm.set_simulation_robot(on_off=False)
        self.xArm.motion_enable(enable=True)
        self.xArm.clean_warn()
        self.xArm.clean_error()
        self.xArm.set_mode(0)
        self.xArm.set_state(0)
        self.xArm.set_servo_angle(angle=self.initPos, wait=True, speed=20, acceleration=0.5, is_radian=False)
        time.sleep(0.1)
        self.realTimeMode()

    def realTimeMode(self):
        self.xArm.set_mode(1)
        self.xArm.set_state(0)

    def probabilityChoice(self):
        choices = list(range(len()))

    def getWeight(self):
        print(self.xArm.joints_torque)

    def fifth_poly(self, q_i, q_f, v_i, vf, t):
        # time/0.005
        traj_t = np.arange(0, t, stepT)
        dq_i = v_i
        dq_f = vf
        ddq_i = 0
        ddq_f = 0
        a0 = q_i
        a1 = dq_i
        a2 = 0.5 * ddq_i
        a3 = 1 / (2 * t ** 3) * (20 * (q_f - q_i) - (8 * dq_f + 12 * dq_i) * t - (3 * ddq_f - ddq_i) * t ** 2)
        a4 = 1 / (2 * t ** 4) * (30 * (q_i - q_f) + (14 * dq_f + 16 * dq_i) * t + (3 * ddq_f - 2 * ddq_i) * t ** 2)
        a5 = 1 / (2 * t ** 5) * (12 * (q_f - q_i) - (6 * dq_f + 6 * dq_i) * t - (ddq_f - ddq_i) * t ** 2)
        traj_pos = a0 + a1 * traj_t + a2 * traj_t ** 2 + a3 * traj_t ** 3 + a4 * traj_t ** 4 + a5 * traj_t ** 5
        for i in range(len(traj_pos)):
            traj_pos[i] = round(traj_pos[i], 5)
        return traj_pos

    def Singlep2ptraj(self, pi, pf, t):
        trajectories = [None] * 7
        for i in range(7):
            trajectories[i] = self.fifth_poly(pi[i], pf[i], 0, 0, t)

        trajectories = np.array(trajectories).T
        return trajectories

    def p2pTraj(self, points):
        pointarray = []
        timearray = []
        soundarray = []
        for p in points:
            pointarray.append(p[0])
            timearray.append(p[1])
            soundarray.append(p[2])
        if self.sim == False:
            IP = self.xArm.angles
        else:
            IP = [0, 0, 0, 0, 0, 0, 0]
        traj = self.Singlep2ptraj(IP, pointarray[0], timearray[0])
        sound = list(np.linspace(0., 0., math.ceil(timearray[0] / stepT)))
        self.movexArm(traj, soundarray[0])
        for i in (range(len(points) - 1)):
            # toAdd =

            traj = self.Singlep2ptraj(pointarray[i], pointarray[i + 1], timearray[i + 1])
            self.movexArm(traj, soundarray[i + 1])
            # if soundarray[i+1] > 0:
            #     sound = sound+ list(np.linspace(0., 1., math.ceil(timearray[i+1]/stepT)))
            # else:
            #     sound = sound + list(np.linspace(0., 0., math.ceil(timearray[i+1]/stepT)))
        # return traj,sound

    def movexArm(self, traj, sound):
        self.client = udp_client.SimpleUDPClient(self.IPtoSEND, self.ports[sound - 1])
        count = 0
        offset = 0.025
        if sound:
            is_timbre = self.ports[sound - 1] // 1000 in (1,)
            if not is_timbre:
                soundarr = np.linspace(0., 1., len(traj))
            else:
                print("FLAG")
                soundarrp1 = np.linspace(0., 1., (int(len(traj) / 2)))
                soundarr = np.concatenate((soundarrp1, np.flip(soundarrp1)))
                while len(soundarr) < len(traj):
                    np.append(soundarr, [0.])
        if sound > 0 and not is_timbre:
            self.client.send_message("/on", 1.)
            print("Setting ON to 1")
        for i in traj:
            start_time = time.time()
            tts = time.time() - start_time
            if self.sim == False:
                self.xArm.set_servo_angle_j(angles=i, is_radian=False)
                # print(i)
                if sound > 0:
                    # send = soundarr[count]
                    print((f"/{self.routes[sound - 1]}", soundarr[count] + offset))
                    self.client.send_message(f"/{self.routes[sound - 1]}", soundarr[count] + offset)
            else:
                print(i)
                if sound > 0:
                    # print(soundarr[count])
                    self.client.send_message(f"/{self.routes[sound - 1]}", soundarr[count] + offset)

            count += 1

            while tts < 0.004:
                tts = time.time() - start_time
                time.sleep(0.0001)
        time.sleep(1)
        if sound > 0 and not is_timbre:
            self.client.send_message("/on", 0.)
            print("Setting ON to 0")

    # def dancevel(self, t):
    #     j_angles = [0, 0, 0, 0, 0, 0, 0]
    #     strike = (math.pi / headS3) * (math.cos((math.pi / headS3) * (t)))
    #     two = (math.pi / headS) * math.cos(math.pi / headS) * (t - 0.75*headS-self.phase*headS)
    #     three = (math.pi / headS3) * (math.cos((math.pi / headS3) * (t-self.phase*headS3)))
    #     four = (math.pi / headS) * math.cos((math.pi / headS) * (t-0.5*headS-self.phase*headS))
    #     five = (math.pi / headS5) * math.cos((math.pi / headS5) * (t-self.phase*headS5))
    #     six = (math.pi / headS) * math.cos((math.pi / headS) * (t-self.phase*headS))
    #     seven = (math.pi / headS5) * math.cos((math.pi / headS5) * (t-self.phase*headS5))

    #     # j_angles[0] = -baseamp * strike
    #     j_angles[1] = -amp1 * two
    #     j_angles[2] = +amptwist * three
    #     j_angles[3] = +amp3 * four
    #     j_angles[4] = +amptwist * five
    #     j_angles[5] = -amp5 * six
    #     j_angles[6] = - amp5 * seven
    #     return j_angles

    # def dancepos(self, t):
    #     j_angles = self.initPos.copy()
    #     strike = (math.sin((math.pi / headS3) * (t)))
    #     two = math.sin((math.pi / headS) * (t - 0.75*headS-self.phase*headS))
    #     three = math.sin((math.pi / headS3) * (t-self.phase*headS3))
    #     four = math.sin((math.pi / headS) * (t-0.5*headS-self.phase*headS))
    #     five = math.sin((math.pi / headS5) * (t-self.phase*headS5))
    #     six = math.sin((math.pi / (headS) * (t-self.phase*headS)))
    #     seven = math.sin((math.pi / (headS5) * (t-self.phase*headS5)))

    #     # j_angles[0] = startp[0] - baseamp * strike
    #     j_angles[1] = self.initPos[1] - amp1 * two
    #     j_angles[2] = self.initPos[2] + amptwist * three
    #     j_angles[3] = self.initPos[3] + amp3 * four
    #     j_angles[4] = self.initPos[4] + amptwist * five
    #     j_angles[5] = self.initPos[5] - amp5 * six
    #     j_angles[6] = self.initPos[6] - amp5 * seven
    #     if abs(j_angles[6]) >= 360:
    #         j_angles[6] = (j_angles[6]/abs(j_angles[6]))*359.8
    #     for i in range(len(j_angles)):
    #         j_angles[i] = round(j_angles[i],5)
    #     return j_angles

    def danceblock(self, tS, tEnd):
        tArr = np.arange(tS, tEnd, stepT)
        blockd = []
        for t in tArr:
            blockd.append(self.dancepos(t))
        return blockd

    def matlabparser(self, traj, startP, v_i, v_f):
        # format is [[[pos1,pos2],[time1,time2]],[[JOINT2pos1,pos2],[time1,time2]]]
        trajblock = []
        j = 0
        for joint in traj:
            # format is now at [[pos1,pos2],[time1,time2]]
            jointblock = np.empty([0, 1])
            if len(joint[0]) == 1:
                iter = 1
            else:
                iter = len(joint[0])
            for i in range(iter):
                if i == 0:
                    v = v_i[j]
                    vf = 0
                    qi = startP[j]
                    qf = joint[0][i]
                elif i == iter - 1:
                    v = 0
                    vf = v_f[j]
                    qi = joint[0][i - 1]
                    qf = joint[0][i]
                else:
                    v = 0
                    vf = 0
                    qi = joint[0][i - 1]
                    qf = joint[0][i]
                timescale = sum(joint[1]) * joint[1][i] / sum(traj[0][1])
                step = self.fifth_poly(qi, qf, v, vf, timescale)
                jointblock = np.append(jointblock, step)
            # print("JOINT BLOCK",j,len(jointblock))
            trajblock.append(jointblock)
            j += 1
        # print(len(trajblock[3]))
        # fix lengths
        lengths = []
        for i in trajblock:
            lengths.append(len(i))
        size = min(lengths)
        # reformat
        reformat = []
        for i in range(size):
            row = []
            # for q in range(7):
            # print(len(trajblock[q]))
            for j in range(7):
                row.append(trajblock[j][i])
            reformat.append(row)
        return reformat

    def makeStrum(self):
        trajectory = []
        for i in range(len(self.tStrums)):
            struminput = self.trajInput[i]
            tCur = self.tStrums[i]
            endT = tCur + sum(struminput[0][1])
            endpos = self.dancepos(endT)
            for j in range(7):
                struminput[j][0][-1] = endpos[j]
            startP = self.dancepos(tCur)
            vstruminit = self.dancevel(tCur)
            vfinal = self.dancevel(endT)
            play = self.matlabparser(struminput, startP, vstruminit, vfinal)
            noplay = []
            tnoplay = np.arange(tCur, endT, stepT)
            for time in tnoplay:
                noplay.append(self.dancepos(time))
            # trajectoryBlock = [play]
            trajectory.append(play)
        return trajectory

    def finalTraj(self):
        # timeArr = self.tStrums
        greenzone = [[0, 0]]
        for i in range(len(self.tStrums)):
            greenzone.append([self.tStrums[i], self.tStrums[i] + 9])
        greenzone.append([self.Period, self.Period])
        # timeArr.append(self.Period)
        interpoints = []
        for i in range(len(greenzone) - 1):
            iterations = []
            toAdd = self.danceblock(greenzone[i][1], greenzone[i + 1][0])
            # for j in range(len(self.trajectories)):
            iterations.append(toAdd)
            interpoints.append(toAdd)

        # interweave in betweens and finals #
        result = [None] * (len(interpoints) + len(self.trajectories))
        result[::2] = interpoints
        result[1::2] = self.trajectories
        strumoutput = []
        for i in result:
            strumoutput = strumoutput + i

        # shift trajectory by phase #
        # the phase factor indicates how much of the trajectory will be put in front of the section
        n = int(self.phase * self.Period / stepT)
        strumresultoffset = strumoutput[-n:] + strumoutput[:-n]

        nostrum = self.danceblock(0, self.Period)
        nostrumoffset = nostrum[-n:] + nostrum[:-n]
        resultoffset = [nostrumoffset, strumresultoffset]
        resultoffset = [nostrum, strumoutput]

        return resultoffset

        # greenzone.append([timeArr[i], timeArr[i]])

    # def createStrumLoop

# r1 = Robots(2,7,4,1,2)
# r1.test2()
# print(r1.startT)
# print(r1.strum8)