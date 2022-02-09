"""
In this file, we implement the plaintiff-law firm-funder contract.
"""

from plaintiff_lawfirm_contract import *

def litigation_funding_contract(S          = 10 ** 6,
                                H          = work_time_distribution( how='empirical_fit' ),
                                W_args     = None,
                                X_L        = 10**6,
                                eta_L      = 0.3,
                                U_args     = {'gamma':0.5, 'kind':'sahara'},
                                ):
    """
    Return optimal funder investment and payoff fraction for a given case. Arguments are the same as for the
    solve_plaintiff_lawyer_game function inside plaintiff_lawfirm_contract.py. See there for a detailed description.

    :return a_F_a:      The optimal hourly rate to be invested into the case by the funder.
    :return eta_F_a:    The optimal payoff received by the founder.
    :return a_L_aa:     The additional hourly rate invested by the law firm.
    :return gain_F:     The funder's relative increase in expected utility from taking the case.
    :return gain_L:     The lawyers's relative increase in expected utility from having the case funded.
    :return gain_P:     The plaintiff's relative increased winning probability from having the case funded.
    """

    # Set utilities and determine what the law firm would invest without third-party funding.
    ####################################################################################################################
    if W_args is None: W_args = { 'a_0':0,  'w_0':0, 'w_1':0.6, 'kappa': 1/250 } # cf. plaintiff-law firm contract
    PL_ret = plaintiff_lawyer_contract(S        = S,
                                       H        = H,
                                       W_args   = W_args,
                                       X_L      = X_L,
                                       eta_L    = eta_L,
                                       U_args   = U_args,
                                       ax       = None,
                                       )

    a_L_a, G_L_a, OO_L = PL_ret['a_L*'], PL_ret['G_L*'], PL_ret['OO_L']

    # Optimize the funder's contract.
    # Note: In a previous implementation, I also had a scipy.minimized based implementation of this optimization.
    # Albeit seemingly more elegant, it is a bit tricky since we have both boundaries and constraints, namely the
    # plaintiff-law firm constraints currently implemented as part of G_F. If we were to use scipy.minimize, we should
    # encode this constraints explicitly as constraints, and not inside G_F (the bounds then also have to be implemented
    # as constraints, since the bounds argument is ignored). Then, another problem was that the objective was still
    # sometimes evaluated with eta_F > 1, despite the constraint being implemented. There are some workarounds, but it
    # is tricky. In the end, I decided to just keep it brute-force, rather than spending many more hours on solving this
    # in a more elegant way.
    # Regarding the brute-force grid, we use the risk-neutral law firm investment as reasonable length scale.
    ####################################################################################################################
    objfunc = partial(  G_F,
                        a_L_a=a_L_a,   G_L_a=G_L_a,   OO_L=OO_L, S=S,  H=H,
                        W_args=W_args, X_L=X_L,       eta_L=eta_L,     U_args=U_args ) # fix all but x=(a, eta)

    x0        = optimal_investment_analytical( S, eta_L, W_args['w_0'], W_args['w_1'], W_args['kappa'], H )
    a_scale   = x0[0] # the set the 'scale' for where the optimal investment lies
    e_scale   = x0[1]

    if a_scale==0:     # special case where the case itself is not lucrative for a risk-neutral investor

        a_F_a     = 0
        eta_F_a   = 0
        obj_evals = None

    else: # generic case

        a_min, a_max   = 0.70  * a_scale, 1.2   * a_scale  # reasonable bounds
        e_min, e_max   = 0.50  * e_scale, 1.70  * e_scale  # cannot ask for less than 0 or more than 1
        e_max          = min(1, e_max)                     # bound it
        a_disc         = int(     10  * (a_max-a_min) )    # discretization up to ten cents
        e_disc         = int(5  * 100 * (e_max-e_min) )    # discretization up to a 5th of a percentage
        a_disc, e_disc = max(5, a_disc), max(5, e_disc)    # special cases

        scales =   [ 'linear', 'linear' ]
        disc   =   {
                    1: [ a_disc,  e_disc ],
                    }

        (a_F_a, eta_F_a), obj_evals  = gridopt(   objfunc         =  objfunc,
                                                  bounds          = [ (a_min, a_max), (e_min, e_max) ],
                                                  discretizations =  disc,
                                                  scale           =  scales,
                                                  minimize        =  False,
                                                  parallel  	  =  True,      # run in parallel
                                                  full_ret        =  True,      # return objective evaluations
                                                  warn            =  True,      # inform if too restrictive boundary
                                                  )

    # Determine the outcomes for all involved parties
    ####################################################################################################################
    G_F_a      = objfunc( [a_F_a, eta_F_a] )                          # expected utility for funder
    p          = W_func( a_F_a, **W_args )                            # winning probability under this contract
    OO_F       = 0                                                    # outside option: funder is risk neutral
    Delta_F    = G_F_a - OO_F                                         # increased utility from investing
    G_L_aa     = law_firm_utility(eta_F_a, p, S, X_L, eta_L, U_args ) # law firm's expected utility
    Delta_L    = G_L_aa - np.nanmax( [G_L_a, OO_L] )                  # gain from third-party funding
    W_L_aa     = W_func( a_F_a,   **W_args )                          # winning probability with funding
    W_L_a      = W_func( a_L_a,   **W_args )                          # winning probability without funding
    Delta_P    = W_L_aa - W_L_a                                       # increased winning probability

    # aggregate the return values
    ####################################################################################################################
    ret = { 'a_F*':a_F_a, 'eta_F*':eta_F_a, 'p':p,
            **PL_ret,
            'W_L**': W_L_aa, 'W_L*': W_L_a,
            'Delta_F':Delta_F, 'Delta_L':Delta_L, 'Delta_P':Delta_P,
            }

    # transform the objective evaluations into a PandasFrame with some additional information for convenience
    ####################################################################################################################
    if obj_evals is not None and len(obj_evals) > 0:

        obj_evals           = pd.DataFrame( obj_evals, columns=['a_F','eta_F','G_F'] )
        obj_evals['a_L*']   = a_L_a
        obj_evals['G_L*']   = G_L_a
        obj_evals['G_L**']  = G_L_aa
        obj_evals['OO_L']   = OO_L

        ret['obj_evals']    = obj_evals

    return ret


def law_firm_utility(eta_F,
                     p, S, X_L, eta_L, U_args ):
    """
    Calculate the law firm's expected utility under a given contract eta_F.
    This is a subroutine of the solve_funder_game, defined with global scope for parallelization. See solve_funder_game
    for details.
    """

    L_gain    = (1-eta_F) * eta_L * S                 # what law firm gets if case is own
    U_L_win   = U_func( X_L + L_gain, **U_args )      # utility of law firm if they win
    U_L_lose  = U_func( X_L,          **U_args )      # utility of law firm if they lose
    G_L_aa    = p * U_L_win  +  (1-p) * U_L_lose      # law firm's expected utility (since 0 cost if they lose)

    return G_L_aa


def G_F( x,
         a_L_a, G_L_a, OO_L, S, H, W_args, X_L, eta_L, U_args ):
    """
    Funder's expected (risk-neutral) utility for a given hourly rate a_F and payoff fraction eta_F; x=(a_F, eta_F).
    This is a subroutine of the solve_funder_game, defined with global scope for parallelization. See solve_funder_game
    for details.
    """

    # Calculate the law firm's expected utility under this contract. If the offer makes the plaintiff worse off (P_cond)
    # or the law firm worse off (L_cond), the contract is not admissible and will be rejected.
    ####################################################################################################################
    a_F, eta_F = x                                              # unpack
    p          = W_func( a_F, **W_args )                        # winning probability under this contract
    G_L_aa     = law_firm_utility(eta_F,p,S, X_L,eta_L,U_args)  # expected utility under this contract
    P_cond     = a_F     < a_L_a                                # if True: plaintiff is worse off
    L_cond     = G_L_aa  < np.nanmax([G_L_a,OO_L])              # if True: law firm is worse off

    if P_cond or L_cond: return np.nan                          # case is rejected

    # Assuming the contract does not make plaintiff or law firm worse off, we can calculate the funders expected utility.
    # For the integration, we apply similar "tricks" as for the plaintiff law firm game. See there for comments.
    ####################################################################################################################
    cost_func =  lambda h: C_func( a_F, h )                    # cost as a function of time
    q         = 0.999                                          # upper quantile to integrate to
    ub        = H.ppf(q)                                       # upper integration boundary
    eps       = 1e-8 * X_L                                     # base value for precision
    prec      = {'epsabs':eps, 'epsrel':1e-12}                 # precision parameters (cf. 2. above)
    Ecost     = H.expect( cost_func, lb=0, ub=ub, **prec ) / q # expected cost
    U_F_win   =  eta_F*eta_L*S - Ecost                         # expected risk-neutral utility if case is won
    U_F_lose  =                - Ecost                         # expected risk-neutral utility if case is lost
    obj       = p * U_F_win  +  (1-p) * U_F_lose               # funder's expected utility under this contract

    return obj
