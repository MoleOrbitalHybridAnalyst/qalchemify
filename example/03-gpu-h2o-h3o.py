import numpy as np
from pyscf import gto
from gpu4pyscf import scf
from gpu4qalchemify import merged_scf_generator

molA = gto.Mole()
molA.atom = \
'''
       O -3.37765e+00  9.76485e-01  2.93603e-01
       H -1.97549e+00  1.41857e+00  5.28212e-01
       H -3.71513e+00  1.69673e-01  7.74414e-01
'''
molA.basis = 'sto-3g'
molA.charge = 0
molA.build()

molB = gto.Mole()
molB.atom = \
'''
       O -3.37765e+00  9.76485e-01  2.93603e-01
       H -1.97549e+00  1.41857e+00  5.28212e-01
       H -3.71513e+00  1.69673e-01  7.74414e-01
       H -4.05053e+00  1.65513e+00  4.50586e-01
'''
molB.basis = 'sto-3g'
molB.charge = 1
molB.build()

v_orb = 500   # max potential on ghost orbitals, in hartree
k_res = 0.01  # in hartree/bohr^2

# f0 controls potential on molB unqiue atom orbitals
def f0(l):
    return np.exp(-20 * l)
def df0(l):
    return -20 * np.exp(-20 * l)

def geom_pred(x, return_grad=False):
    '''
    x would be the coords (in Bohr) of shared atoms
    i.e., the first three atoms (O,H,H) in this case
    '''
    center = (2*x[0]-x[1]-x[2])/1.2+x[0]
    # g[i,x,j,y] = d center[i, x] / d x[j, y]
    g = np.zeros((1,3,3,3))
    g[0,:,0,:] = np.eye(3) * (2/1.2+1)
    g[0,:,1,:] = np.eye(3) * (-1/1.2)
    g[0,:,2,:] = np.eye(3) * (-1/1.2)
    if return_grad:
        return g
    else:
        return center

def rhf(mol):
    mf = scf.hf.RHF(mol)
    mf.init_guess = '1e'                                                                  # interpolation makes default minao guess makes less sense, use 1e instead
    return mf
mfgen = merged_scf_generator(
        rhf,                                                                              # will create a restricted Hartree Fock object
        molA, molB,                                                                       # molA -> molB when lambda = 0 -> 1
        [0,1,2],                                                                          # indices of molA atoms whose coords will be shared with molB
        [0,1,2],                                                                          # indices of molB atoms whose coords will be shared with molA
        [],                                                                               # indices of molA unique atoms
        [3],                                                                              # indices of molB unique atoms
        has_grad=True,                                                                    # indicates the input switching functions can return gradients
        fsw_nelectron=None,                                                               # switching function to transform the number of electrons
        fsw_spin=None,                                                                    # switching function to transform the spin value
        fsw_ham_single=None,                                                              # switching function to transform the Hamiltonian for shared coord atoms
        fsw_ham_dualA=None,                                                               # switching function to turn off the Hamiltonian for molA unique atoms
        fsw_ham_dualB=lambda l, return_grad=False: [1-l,-1][return_grad],                 # switching function to turn off the Hamiltonian for molB unique atoms
        vorb_molA=None,                                                                   # potential on molA unqiue atom orbitals to penalize population
        vorb_molB=lambda l, return_grad=False: [v_orb*f0(l), v_orb*df0(l)][return_grad],  # potential on molB unique atom orbitals to penalize population
        geom_res_fc_dualA=None,                                                           # geometric restraint force constant for molA unique atoms
        geom_res_fc_dualB=lambda l, return_grad=False: [k_res*(1-l),-k_res][return_grad], # geometric restraint force constant for molB unique atoms
        geom_dualA_pred=None,                                                             # geometric restraint center for molA unique atoms
        geom_dualB_pred=geom_pred                                                         # geometric restraint center for molB unique atoms
        )


# check we do have h2o when lambda = 0.0
mf0 = scf.hf.RHF(molA)
e0 = mf0.kernel()
mf0grad = mf0.nuc_grad_method()
gradR0 = mf0grad.kernel()
mf = mfgen(0.0)
e = mf.kernel() - mf.mol.energy_geom_res()
mfgrad = mf.nuc_grad_method()
gradR = mfgrad.kernel() - mf.mol.grad_geom_res()                                          # subtract the geometric restraint energy which is absent in a regular water
assert abs(e0 - e) < 1e-4
assert np.max(np.abs(gradR0 - gradR[:3])) < 1e-4                                          # gradR0 only has the gradient of molA atoms while gradR has the additional molB extra proton
assert np.max(np.abs(gradR[3])) < 1e-4                                                    # check extra proton has no force when in the "h2o" state

# check we do have h3o+ when lambda = 1.0
mf1 = scf.hf.RHF(molB)
e1 = mf1.kernel()
mf1grad = mf1.nuc_grad_method()
gradR1 = mf1grad.kernel()
mf = mfgen(1.0)
e = mf.kernel()
mfgrad = mf.nuc_grad_method()
gradR = mfgrad.kernel()
assert abs(e1 - e) < 1e-4
assert np.max(np.abs(gradR1 - gradR)) < 1e-4

# check lambda gradient using finite difference
def check_lambda_gradient(l0):
    dEdl = mfgen(l0).run().energy_tot_lgrad()
    dEdlnum = (mfgen(l0+1e-5).kernel() - mfgen(l0-1e-5).kernel()) / 2e-5
    assert abs(dEdl - dEdlnum) < 1e-5
check_lambda_gradient(1.00)
check_lambda_gradient(0.95)
check_lambda_gradient(0.50)
check_lambda_gradient(0.05)
check_lambda_gradient(0.00)
