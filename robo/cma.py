
def _test():
    """
    import barecmaes2 as cma
    import random
    random.seed(5)
    x = cma.fmin(cma.Fcts.rosenbrock, 4 * [0.5], 0.5, verb_plot=0)
    evals: ax-ratio max(std)   f-value
        8:     1.0  4.3e-01  8.17705253283
       16:     1.1  4.2e-01  66.2003009423
      800:    19.1  3.8e-02  0.0423501042891
     1600:    46.1  4.7e-06  1.14801950628e-11
     1736:    59.2  5.4e-07  1.26488753189e-13
    termination by {'tolfun': 1e-12}
    best f-value = 4.93281796387e-14
    solution = [0.9999999878867273, 0.9999999602211, 0.9999999323618144, 0.9999998579200512]

  """

    import doctest
    print('launching doctest')
    doctest.testmod(report=True)  # module test
    print("""done (ideally no line between launching and done was printed,
        however slight deviations are possible)""")

#_____________________________________________________________________
#_____________________________________________________________________
#
if __name__ == "__main__":

    _test()

    # fmin(Fcts.rosenbrock, 10 * [0.5], 0.5)