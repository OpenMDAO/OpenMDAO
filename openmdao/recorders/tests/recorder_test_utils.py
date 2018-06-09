import time

def run_driver(problem, **kwargs):
    t0 = time.time()
    problem.run_driver(**kwargs)
    t1 = time.time()
    return t0, t1

