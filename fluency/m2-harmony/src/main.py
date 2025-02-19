import utils
from melody import MelodyGenerator

if __name__ == "__main__":
    simulation = True
    xarm = utils.Robot("192.168.1.215", sim=simulation)
    if not simulation:
        xarm.setupBot()
    mg = MelodyGenerator(scale='major')
    # xarm.from_file('resources/demo.txt', 4)
    # xarm.from_file(mg.generate(length=1), 5)
    # xarm.harmony(5)
    xarm.from_curve((0.1, 0.3, 0.7, 0.9))