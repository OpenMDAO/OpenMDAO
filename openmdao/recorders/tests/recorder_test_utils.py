import time

def run_driver(problem):
    t0 = time.time()
    problem.run_driver()
    t1 = time.time()
    return t0, t1

