import utils

if __name__ == "__main__":
    simulation = True
    xarm = utils.Robot("192.168.1.215", simulation)
    if not simulation:
        xarm.setupBot()
    xarm.from_file('resources/demo.txt', 4)
