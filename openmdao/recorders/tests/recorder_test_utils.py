import time

def run_driver(problem, **kwargs):
    t0 = time.perf_counter()
    problem.run_driver(**kwargs)
    t1 = time.perf_counter()
    return t0, t1

