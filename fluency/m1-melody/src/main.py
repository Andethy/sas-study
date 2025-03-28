import numpy as np
import math
import time
# from xarm.wrapper import XArmAPI
import utils
import random

# mapping = {'H1': 1, 'M1': 8, 'S1': 2}

types = "HMST"  # HMST
div = len(types)
per = 4
mapping = {f'{types[i // per]}{i % per + 1}': (i + 1) for i in range(div * per)}


# Autogenerated -> {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'M1': 5, 'M2': 6, 'M3': 7, 'M4': 8, 'S1': 9, 'S2': 10, 'S3': 11, 'S4': 12}


def main(xarm):
    # demosMelody = ['M1','M2','M3','M4']
    # demosScale = ['S1','S2','S3','S4']
    # demosSilence = ['H1']

    # demosTimbre = ['T1', 'T2', 'T3', 'T4']
    demosMelody = ['M1', 'M2', 'M3', 'M4']
    demosScale = ['S1', 'S2', 'S3', 'S4']
    demosSilence = ['n', 'n', 'n', 'n']
    options = [demosScale]
    # options = [demosMelody,demosScale,demosSilence]
    demos = []

    for demo in options:
        # print(demo[0])

        test = list(demo.copy())
        random.shuffle(test)
        test2 = list(demo.copy())
        random.shuffle(test2)
        demos.append(test + test2)
    neworder = demos.copy()
    random.shuffle(neworder)
    print(neworder)
    neworderp2 = neworder.copy()
    random.shuffle(neworderp2)
    neworder += neworderp2
    print(neworder)
    # input("press Enter to start")
    # a
    for demos in neworder:
        input("Ready to start")
        # demos = list(mapping)
        # random.shuffle(demos)
        # d2 = demos.copy()
        # random.shuffle(d2)
        # demos += d2
        # print(demos)

        # demos = ['S1','S2','S3','S4']
        # random.shuffle(demos)
        # d2 = demos.copy()
        # random.shuffle(d2)
        # demos += d2
        count = 0
        speed = list(np.linspace(3.5, 6.6, 4))
        random.shuffle(speed)
        speed += speed
        delay = list(np.linspace(1, 5, 4))
        random.shuffle(delay)
        delay += delay
        print(len(delay))
        # input("dfgdgf")
        for demo in demos:
            # if demo[0] in ('H', 'T'):
            #     continue
            # demo = 'M2'
            if demo == 'n':
                sound = 0
            else:
                sound = mapping[demo]
            print(sound)

            t = 3 * random.random() + 3
            traj_option = [[95, 15, 30, 85, -23, -28, 0], [108, 50, 10, 100, -11, -35, 0], [84, 25, 15, 100, -12, 0, 0]]
            traj_choice = random.randrange(2)
            print(traj_choice)
            # delay = random.randrange(0, 5)
            time.sleep(delay[count])
            print("Doing trajecrtory " + demo)
            points = [[[0, 35, 15, 125, -23, 0, 0], 3, 0], [traj_option[traj_choice], speed[count], sound],
                      [[0, 35, 15, 125, -23, 0, 0], 2, 0], [[0, 35, 15, 125, -23, 30, -180], 1, 0]]
            xarm.p2pTraj(points)
            count += 1


if __name__ == "__main__":
    simulation = True
    xarm = utils.StaticRobot("192.168.1.215", simulation)
    if not simulation:
        xarm.setupBot()
    input("hi")
    main(xarm)

    # sound = -1
    # while sound:
    #     sound = int(input("input sound ID to move: "))
    #     t = 5
    #     if not sound:
    #         break
    #     points = [ [[586,17.9,689,179.6,-71,2],3,0], [[-57,571,456,-160,-39,88],t,sound],[[586,17.9,689,179.6,-71,2],t,sound]] #0,17.6,0,115,0,26  92,11,5,30,30,-18
    #     xarm.p2pTraj(points)
    # print(trajectory[0])  