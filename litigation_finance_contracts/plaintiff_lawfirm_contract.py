"""
In this file, we implement the plaintiff-law firm contract, along with some other basic functions that have been used
throughout the paper.
"""

import numpy   as np
import pandas  as pd
import seaborn as sb

from functools      import partial
from scipy.stats    import lognorm, gamma
from scipy.optimize import minimize

from numpy import exp, sqrt, log

# repo is available here: https://github.com/lerasc/hierarchical_grid_optimization
from hierarchical_grid_opt.grid_optimization import grid_optimization as gridopt


def U_func( X, gamma=None, kind='sahara' ):
    """
    Utility function for law firm.

    :param X:           Wealth.

    :param gamma:       Risk-neutrality parameter.

    :param kind:        What Type of functional shape is used:
                        - 'risk-neutral':   just the identify function, here gamma is ignored
                        - 'exp':            exponential utility, gamma is decay parameter
                        - 'power':          power-utility, with gamma as power exponent
                        - 'log':            log-utility (gamma is ignored)
                        - 'sahara':         Sahara utility, see definition inside the paper.
    """

    # simple case: risk-neutrality
    if   kind=='risk_neutral': return  X

    # the following utilities are common but not defined for negative wealth:
    elif kind=='exp':          return  exp( -gamma*X )
    elif kind=='power':        return  X**gamma / gamma
    elif kind=='log':          return  log( X )

    # The Sahara utility function generalizes the power-utility on negative numbers. See Equation (2) in [1] for the
    # exact formula.
    # [1] 2015 - Chen et al. - Modeling non-monotone risk aversion using SAHARA utility functions
    elif kind=='sahara':

        beta = 10                                                                     # fix for simplicity
        SQ   = sqrt( beta**2 + X**2 )                                                 # for convenience

        if gamma==0.0:   return X                                                     # for faster execution
        if gamma==1.0:   return 0.5 * log( X + SQ ) + 0.5 * X * ( SQ - X ) / beta**2  # special case ~ log
        else:            return - (X + SQ)**(-gamma) * (X + gamma*SQ) / (gamma**2-1)  # generic case

    else:

        raise ValueError('invalid argument kind=%s'%kind)


def W_func( a, a_0=0, w_0=0, w_1=0.8, kappa=1 ):
    """
    Parametrized probability to win a case as a function of hourly law firm rate.

    :param a:           hourly law firm rate
    :param a_0:         minimum hourly rate to be charged
    :param w_0:         probability to win a case if a_0 is invested
    :param w_1:         probability to win a case if an infinite amount is invested
    :param kappa:       scaling parameter
    """

    p =  1 - np.exp( - kappa * (a-a_0)  )
    p = (w_1-w_0) * p + w_0

    return p


def C_func( a, H, c_0=0 ):
    """
    Process cost as a function of number of work hours H.

    :param a:       hourly law firm rate
    :param H:       number of hours worked on the case
    :param c_0:     time independent cost
    """

    return c_0  +  a * H


def work_time_distribution( how='gamma' ):
    """
    Return a scipy stats objected that represents the duration of case time in units of hours worked on the case H.

    :param how:    what type of distribution to use:

                    - 'gamma':      return a Gamma distribution, with parameters fitted with empirical data (cf. paper).

                    - 'lognormal':  return a log-normal distribution, with parameters fitted with empirical data.
                                    Attention: the log-normal produces a much too heavy tail. Not recommended to use


    :return:       scipy.stats.rv_continuous object
    """

    # We consider the empirical distribution of case durations in units of days. Then, we transform to hours by assuming
    # a 1 hour per day. We have fitted a gamma distribution to the empirical data, and found the parameters implemented 
    # below. See paper for the definition of the gamma pdf with parameters a and b.
    if how=='gamma':

        a       = 1.04                               # parameter 'a', like in paper
        loc     = 0.00                               # translational shift, not needed here, but included in scipy.
        scale   = 212.77                             # inverse of parameter 'b' (cf. paper)
        dist    = gamma( a=a, loc=loc, scale=scale ) # initialize instance
 
    # Same as for 'gamma' but with a log-normal distribution. In practice, it is not a good choice since it creates a
    # right tail that is much fatter than the empirical one. This causes huge tail costs that lead to rejection of 
    # most cases.
    elif how=='lognormal':

        mu         = 4.83
        sigma      = 1.12
        dist       = lognorm( s=sigma, scale=np.exp(mu) )

    else:

        raise ValueError('invalid optimizer argument')

    return dist


def optimal_investment_analytical( S, H, eta_L=0.3, w0=0, w1=0.7, kappa=1/250 ):
    """
    Under the assumption of a risk-neutral law firm, we can determine analytically the amount to be invested as well as
    the fraction to be obtained by the law firm (see paper). We use this function to determine optimal starting values
    in investments.

    :param S:           case payoff
    :param H:           scipy stats instance of work hour random variable H
    :param eta_L:       fraction of the payoff received by the law firm
    :param w0:          minimal winning probability
    :param w1:          maximum winning probability
    :param kappa:       sensitivity of winning probability to invested effort (cf. W_func)
    """

    E     = H.expect( lambda x: x )
    y     = ( kappa * (w1-w0) * eta_L * S) / E
    y     = max(y, 1e-10) # regularization

    a_th  = np.log(y)  / kappa # optimal amount to invest
    a_th  = max(0,a_th)

    e_th  = E * np.log(y) / (kappa * w1 * eta_L * S - E) # optimal fraction to ask for
    e_th  = max(0,e_th)

    return a_th, e_th


def plaintiff_lawyer_contract(S           = 10**5,
                              H           = work_time_distribution( how='gamma' ),
                              W_args      = None,
                              X_L         = 10**6,
                              eta_L       = 0.3,
                              U_args      = {'gamma':0.5, 'kind':'sahara'},
                              optimizer   ='brute_force',
                              ax          = None,
                              ):
    """
    Return optimal law firm investment for a given case.

    :param S:           case payoff

    :param H:           scipy stats instance of work hour random variable H

    :param W_args:      dict of arguments for the W function (cf. W_func, fixed internally if set to None)

    :param X_L:         law firm wealth

    :param eta_L:       fraction of the payoff received by the law firm

    :param U_args:      dict of arguments for the law firm's utility function (cf. U_func)

    :param optimizer:   how to optimize ('scipy' means gradient descent, 'brute_force' means grid-evaluation)

    :param ax:          if not None, an axis instance in which the result is to be visualized

    :return a_L_a:      the optimal fee that is to be invested into the case (between a_0 and a_1)

    :return OJ:         the expected utility when taking on the case

    :return OO:         the utility when not taking the case, U(X_L), i.e. the outside option
    """

    # implement the law firm's objective function
    ####################################################################################################################
    def U_L( h, a, case='win' ):
        """
        Law firm's utility if the case is 'won' (or 'lost') at time t with invested cost a. (We don't use lambda if
        functions, so that we can parallelize.)
        """

        if   case=='win':   val = U_func( X_L + eta_L * S - C_func(a, h), **U_args )
        elif case=='lose':  val = U_func( X_L             - C_func(a, h), **U_args )
        else:               raise ValueError('invalid case argument')

        val = max(val, -10**9) # regularization for large losses (to avoid NaN in integration)

        return val

    def G_L( a ):
        """
        Lawyer objective function for the plaintiff-law firm contract.
        """

        # It turned out that the (naive) numerical integration becomes unreliable for two distinct reasons:
        # 1.    The integration to infinity is not well handeled. We have checked that integrating up to a large
        #       quantile does not affect the integral value in a significant manner. We thus integrate only up to a
        #       some upper bound (this also makes the numerical integration much faster).
        #       However, to assert that the integral is still normalized (since we weight by a probability), we have to
        #       renormalize.
        # 2.    The default scipy.integrate.quad precission (on which H.expect() is built) is set to 1e-8. However,
        #       The Sahara utility becomes very small for gamma > 1. So for instance, the utility of 1 million is
        #       about 10^{-14} for a value of gamma=2. Therefore, we must set the quad default tolerance dynamically.
        #       Specifically, we set it orders of magnitude lower than the utility of X_L.


        p        = W_func(a, **W_args)                                     # probability to win case
        U_win    = partial( U_L, a=a, case='win'  )                        # utility to win a  case
        U_lose   = partial( U_L, a=a, case='lose' )                        # utitlity to lose a case
        q        = 0.999                                                   # upper quantile to integrate to
        ub       = H.ppf(q)                                                # upper integration boundary
        U_base   = U_lose( h=0 )                                           # utility of X_L
        eps      = 1e-8 * abs(U_base)                                      # make it much smaller
        prec     = {'epsabs':eps, 'epsrel':1e-12}                          # precision parameters (cf. 2. above)
        E_win    = H.expect(U_win ,  lb=0,  ub=ub, **prec ) / q            # expected gain (renormalized, cf. 1. above)
        E_lose   = H.expect(U_lose , lb=0,  ub=ub, **prec ) / q            # expected gain (renormalized, cf. 1. above)
        obj      = p * E_win   +   (1-p) * E_lose                          # expected value

        return obj

    # Find the optimal hourly rate. If W_args is none, we set it to a heurstically reasonable value. To set the scale
    # of the search range, we orient ourselves on the analtically known, optimal investment of a risk-neutral funder.
    ####################################################################################################################
    if W_args is None: W_args = {'a_0':0,  'w_0':0, 'w_1':0.6, 'kappa': 1/250 }

    x0        = optimal_investment_analytical( S, H, eta_L, W_args['w_0'], W_args['w_1'], W_args['kappa'] )
    a_scale   = x0[0]         # the set the 'scale' for where the optimal investment lies
    a_min     = 0             # respect lower bound, if any
    a_max     = 1.2 * a_scale # realistic upper bound (only used if gamma < 0. If gamma > 0, will never invest more

    if optimizer== 'brute_force':

        disc         = int(0.1*(a_max-a_min))                                 # discretization up to 10 dollars
        disc         = max(5, disc)                                           # special case
        disc         = { 1:[disc], 2:[10], 3:[10] }                           # discretization up to 10 cents
        a_L_a, eval  = gridopt(	objfunc         =   G_L,
                                bounds          = [ (a_min, a_max) ],
                                discretizations =   disc,
                                scale           =  'linear',
                                minimize        =   False,                     # we want the maximum
                                parallel		=   False,                     # funder contract calls this in parallel
                                full_ret        =   True,                      # for plotting
                                )
        a_L_a = a_L_a[0]                                                       # since it is 1d

    elif optimizer== 'scipy':

        res = minimize(  fun     = lambda x: -G_L(x), # negative, to minimize
                         x0      = 0.5*a_scale, # pick the optimal investment if the funder was risk-neutral
                         tol     = 1e-4 * U_L( h=0, a=0, case='lose' ), # realistic scale (cf. G_L documentation)
                         bounds  = [(0, a_max)],
                         method  = 'SLSQP',
                        )

        a_L_a = res.x[0] # unpack the result

        if not res.success or res.nit < 2:
            print(f"Failed to converge for S={S}, X_L={X_L}, gamma={U_args['gamma']}, W_1={W_args['w_1']}.")
            print(f"start value:{x0[0]}")
            print(f"Log:\n{res}")

    else:

        raise ValueError('invalid optimizer argument')

    # Given the optimum, analyze what is the law firm's expected utility gain.
    ####################################################################################################################
    OJ  = G_L( a_L_a )                              # utility when taking the case
    OO  = U_func(X_L, **U_args)                     # outside option: don't take the case

    ret = {'a_L*':a_L_a, 'G_L*':OJ, 'OO_L':OO }     # return information

    if ax is None: return ret                       # if no plot, then return

    # Evaluate objective function
    ####################################################################################################################
    if optimizer=='scipy': # need to calculate evaluation points

        als    =   np.linspace( a_min, a_max, num=100 )
        ps     = [ W_func(a, **W_args) for a in als ]
        gs     = [ G_L(a)              for a in als ]
        eval   =   np.array([ als, gs, ps ]).T
        eval   =   pd.DataFrame( eval, index=als, columns=['a', 'G_L', 'w'] )

    else:

        eval      = pd.DataFrame(eval, columns=['a','G_L'])
        eval      = eval.sort_values(by='a', ascending=True)
        eval['w'] =  [ W_func(a, **W_args) for a in eval['a'] ]

    eval['G_L'] = eval['G_L'].clip( lower=-10 )


    # Visualize the objective function G_L(alpha) and its optimum.
    ####################################################################################################################
    sb.set_style('whitegrid')
    axt  = ax.twinx()
    fs   = 15
    cols = sb.color_palette('bright',2)

    ax.axvline( x=a_L_a, linestyle='--', color='grey',  linewidth=3,
                label=r'$\alpha_L^*=$' + f'{int(a_L_a):,}'
                )

    ax.plot( eval['a'], eval['G_L'], color=cols[0], linewidth=2,
             # label=r'$G_L$ with $G(\alpha_L^*)=$' + f'{int(OJ):,}'
             label=r'$G_L$',
             )

    ax.axhline( y=OO, linestyle=':',  color=cols[0], linewidth=3,
                # label=r'$U_L(X_L)=$' + f'{int(OO):,}'
                label=r'$U_L(X_L)$',
                )

    axt.plot( eval['a'], eval['w'], color=cols[1], linewidth=2, linestyle='--')

    # sb.rugplot( eval['a'], color='grey', ax=ax )

    # ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$',      fontsize=fs )
    ax.set_ylabel(r'$G_L$',         fontsize=fs, color=cols[0])
    axt.set_ylabel(r'$w(\alpha)$',  fontsize=fs, color=cols[1])

    # set the axes colors and legend
    ####################################################################################################################
    axes   = [ax, axt]
    colors = [cols[0], cols[1]]
    locs   = ['left', 'right']

    for (ax,color, loc) in zip( axes, colors, locs ):

        ax.spines[loc].set_color(color)
        ax.set_ylabel(ax.get_ylabel(),  color=color, fontsize=fs)
        _ = [t.set_color(color) for t in ax.xaxis.get_ticklines()]
        _ = [t.set_color(color) for t in ax.xaxis.get_ticklabels()]
        _ = [t.set_color(color) for t in ax.yaxis.get_ticklines()]
        _ = [t.set_color(color) for t in ax.yaxis.get_ticklabels()]

    ax.legend(loc='lower center', frameon=False, ncol=1, fontsize=fs-1)
    axt.grid(False)

    return ret, eval
