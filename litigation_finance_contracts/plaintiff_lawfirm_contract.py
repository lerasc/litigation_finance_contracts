"""
In this file, we implement the contract between the plaintiff and the law firm.
"""

import numpy   as np
import pandas  as pd
import seaborn as sb

from functools import partial
from numpy     import exp, sqrt, log

# repo is available here: https://github.com/lerasc/hierarchical_grid_optimization
from hierarchical_grid_opt.grid_optimization import grid_optimization as gridopt


def W_func( a, a_0=0, a_1=None, w_1=0.8, kappa=1, kind='linear' ):
    """
    Parametrized probability to win a case as a function of hourly law firm rate.

    :param a:           hourly law firm rate
    :param a_0:         minimum hourly rate to be charged
    :param a_1:         upper bound above which the effort no more improves the probability to win (if kind='linear')
    :param w_1:         probability to win a case if an infinite amount is invested    
    :param kappa:       scaling parameter (if kind !='linear')
    :param kind:        what type of function to consider: 'linear', 'sigmoid' or 'power'. 
    """

    if kind=='linear':

        p = w_1 * (a - a_0) / (a_1 - a_0)
        p = np.clip(  p, a_min=0, a_max=w_1  )

    elif kind=='sigmoid':

        p = w_1 * ( 1 - np.exp( - kappa * (a-a_0) ) )

    elif kind=='power': raise NotImplementedError(f'implement power-function here')
    else:               raise ValueError(f'invalid  type of function {kind}')

    return p


def plaintiff_lawyer_contract(  R           = 10**5,
                                EH          = 1, 
                                VH          = 1, 
                                lam         = 1.0,                                
                                eta_L       = 0.3,
                                W_args      = {},
                                c0          = 0, 
                                ax          = None ):
    """
    Return optimal law firm investment for a given case.

    :param R:           case payoff
    :param EH, VH:      expected case duration and variance thereof
    :param lam:         parameter of risk aversion lambda. 
    :param eta_L:       fraction of the payoff received by the law firm
    :param W_args:      dict of arguments for the W function (cf. W_func, fixed internally if set to None)           
    :param c0:          fixed costs for this case
    :param ax:          if not None, an axis instance in which the result is to be visualized

    :return a_L_a:      the optimal fee that is to be invested into the case (between a_0 and a_1)
    :return OJ:         the expected utility when taking on the case
    """

    # implement the law firm's objective function
    ####################################################################################################################
    def G_L( a ):
        """
        Lawyer objective function for the contract with the plaintiff. 
        """

        Rt  = eta_L * R                                 # actual reward that law firm obtains
        w   = W(a)                                      # probability to win with this effort
        E   = Rt*w - ( a*EH + c0 )                      # expected gain 
        V   = Rt**2 ( w - w**2 ) + a**2 * VH            # variance of gain
        obj = E - 0.5 * lam * V                         # objective function: mean-variance trade-off

        return obj

    # Find the optimal hourly rate.
    ####################################################################################################################
    W            = partial( W_func, **W_args )                     # probability to win the case for given effort "a"
    Rt           = eta_L * R                                       # actual reward that law firm obtains
    a_min        = 0                                               # lowest possible investment
    a_max        = Rt                                              # maximum investment (= payoff if case is won)
    disc         = int(0.1*(a_max-a_min))                          # discretization up to 10 dollars
    disc         = max( 5, disc )                                  # special case
    disc         = { 1:[disc], 2:[10], 3:[10] }                    # discretization up to 10 cents
    a_L_a, eval  = gridopt(	objfunc         =   G_L,
                            bounds          = [ (a_min, a_max) ],
                            discretizations =   disc,
                            scale           =  'linear',
                            minimize        =   False,                     # we want the maximum
                            parallel		=   False,                     # funder contract calls this in parallel
                            full_ret        =   True,                      # for plotting
                            )

    # Given the optimum, analyze what is the law firm's expected utility gain.
    ####################################################################################################################
    OJ  = G_L( a_L_a )                              # utility when taking the case

    ret = {'a_L*':a_L_a, 'G_L*':OJ }                # return information

    if ax is None: return ret                       # if no plot, then return

    # Create DataFrame with information about evaluation points
    ####################################################################################################################
    eval        =    pd.DataFrame(eval, columns=['a','G_L'])
    eval        =    eval.sort_values(by='a', ascending=True)
    eval['w']   =  [ W(a) for a in eval['a'] ]
    eval['G_L'] =    eval['G_L'].clip( lower=-10 ) # for nicer plotting

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

    axt.plot( eval['a'], eval['w'], color=cols[1], linewidth=2, linestyle='--')

    # sb.rugplot( eval['a'], color='grey', ax=ax )

    # ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$',      fontsize=fs )
    ax.set_ylabel(r'$G_L$',         fontsize=fs, color=cols[0])
    axt.set_ylabel(r'$w(\alpha)$',  fontsize=fs, color=cols[1])
    ax.legend(loc='lower center', frameon=False, ncol=1, fontsize=fs-1)

    # set the axes colors and legend
    ####################################################################################################################
    axes   = [ax, axt]
    colors = [cols[0], cols[1]]
    locs   = ['left', 'right']

    for (ax,color, loc) in zip( axes, colors, locs ):

        ax.spines[loc].set_color(color)
        ax.set_ylabel(ax.get_ylabel(),  color=color, fontsize=fs)
        ax.yaxis.get_offset_text().set_color(color)
        _ = [t.set_color(color) for t in ax.yaxis.get_ticklines()]
        _ = [t.set_color(color) for t in ax.yaxis.get_ticklabels()]

    axt.grid(False)

    return ret, eval
