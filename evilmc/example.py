import numpy as np
import matplotlib.pyplot as plt

from .core import EVILMC_plphs

__all__ = ['example_EVILMC']


def example_EVILMC():
# ;+
# ; NAME:
# ;	example_EVILMC
# ;
# ; PURPOSE:
# ;	Provides an example system (HAT-P-7) to use to the EVIL-MC model. Code is provided with no
# ;	warranties whatsoever at http://www.lpl.arizona.edu/~bjackson/code/idl.html. See also
# ;	paper Jackson et al. (2012) ApJ 750, 1 for more details.
# ;
# ; CATEGORY:
# ;	Astrophysics.
# ;
# ; CALLING SEQUENCE:
# ;	example_EVILMC
# ;
# ; INPUTS:
# ;	none
# ;
# ; OUTPUTS:
# ;	none
# ;
# ; SIDE EFFECTS:
# ;	Makes a plot of the normalized ellipsoidal variation on the screen
# ;
# ; EXAMPLE:
# ;	example_EVILMC
# ;
# ; MODIFICATION HISTORY:
# ; 	Written by:	Brian Jackson (decaleus@gmail.com), 2012 April 18.
# ;-

    # ;These are the model parameters used in Jackson et al. (2012) ApJ 750, 1.
    q = 1.10e-3
    semi = 4.15 #;A/R_0
    per = 2.204733 #;days
    Omega = 4.73e-7 #;s^{-1}
    Ts = 6350. #;K
    Kz = 300. #;m/s
    logg = 4.07 #;log(cm/s^2)
    M = 0.26 #;[Fe/H]
    ecc = 0.0 #;orbital eccentricity
    asc_node = 0. #;longitude of planetary ascending node
    peri_long = 0. #;longitude of planetary pericenter
    inc = 83.1 #;orbital inclination in degrees
    bet = 0.0705696 #;gravity darkening exponent, (T/T_0) = (g/g_0)^\beta
    #;limb-darkening coefficients, I(\mu)/I(1) = 1 - \gamma_1 (1-\mu) - \gamma_2 (1-\mu)^2
    gam1 = 0.314709
    gam2 = 0.312125
    gam = [gam1, gam2]

    #;number of latitude or longitude grid points on stellar surface
    num_grid = 31

    #;orientation (X, Y, Z) of stellar rotation axis
    Omegahat = np.array([0., 0., 1.0])
    #;stellar rotation axis
    Omega = Omega*Omegahat

    # ;planetary phase curve parameters
    # ;
    # ;secondary eclipse depth
    D = 61e-6
    F1 = 30e-6
    F0 = D - F1

    p = [num_grid, q, Kz, Ts, Omega, gam, bet, semi, per, inc, ecc, asc_node, peri_long, F0, F1]

    # ;orbital phase
    num_phs = 10001
    phs = np.arange(num_phs)*1./(num_phs-1.)

    # ;call EVIL-MC routine, along with planetary phase function
    ret = EVILMC_plphs(phs, p)

    plt.plot(phs, ret)
    plt.ylim([min(ret), max(ret)])
    plt.show()

