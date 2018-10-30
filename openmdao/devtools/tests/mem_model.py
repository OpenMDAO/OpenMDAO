from openmdao.api import Problem
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup


def main():
    E = 1.
    L = 1.
    b = 0.1
    volume = 0.01

    num_elements = 50 * 32
    num_cp = 4
    num_load_cases = 32

    prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                             num_elements=num_elements, num_cp=num_cp,
                                             num_load_cases=num_load_cases))
    prob.setup()
    prob.run_model()

main()

