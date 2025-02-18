import numpy as np
from pyscf import gto
from gpu4pyscf import scf
from gpu4qalchemify import merged_scf_generator

molA = gto.Mole()
molA.atom = \
''' 
C           -1.42203        2.29540       -0.02117
H           -1.09131        1.75525       -0.90230
H           -1.06708        1.77522        0.85454
H           -2.50707        2.30263       -0.00361
O           -0.86670        3.62335        0.04489
H           -1.14214        4.14086       -0.72261
''' 
molA.basis = 'sto-3g'
molA.build()

molB = gto.Mole()
molB.atom = \
''' 
C           -1.42203        2.29540       -0.02117
H           -1.09131        1.75525       -0.90230
H           -1.06708        1.77522        0.85454
H           -2.50707        2.30263       -0.00361
F           -0.86670        3.62335        0.04489
''' 
molB.basis = 'sto-3g'
molB.build()

v_orb = 500   # max potential on ghost orbitals, in hartree
k_res = 0.01  # in hartree/bohr^2

# f0 controls potential on molB changing atom (F) orbitals
def f0(l):
    return np.exp(-20 * l)
def df0(l):
    return -20 * np.exp(-20 * l)
# f1 controls potential on molA changing atom (O,H) orbitals
def f1(l):
    return np.exp(-20 * (1-l))
def df1(l):
    return 20 * np.exp(-20 * (1-l))

def geom_pred(x, return_grad=False):
    center = (x[4]-x[0])/1.4 + x[4]
    grad = np.zeros((1,3,5,3))
    grad[0,:,0,:] = -1/1.4 * np.eye(3)
    grad[0,:,4,:] = (1/1.4 + 1) * np.eye(3)
    if return_grad:
        return grad
    else:
        return center

def rhf(mol):
    mf = scf.RHF(mol)
    mf.init_guess = '1e'
    return mf
mfgen = merged_scf_generator(rhf, molA, molB, [0,1,2,3,4], [0,1,2,3,4], [5], [],
        has_grad=True,
        fsw_nelectron=None, fsw_spin=None,
        fsw_ham_single=lambda l, g=False: [1-l,-1][g],
        fsw_ham_dualA=lambda l,g=False: [1-l,-1][g],
        fsw_ham_dualB=None,
        vorb_molA=lambda l,g=False: [v_orb*f1(l),v_orb*df1(l)][g],
        vorb_molB=lambda l,g=False: [v_orb*f0(l),v_orb*df0(l)][g],
        geom_res_fc_dualA=lambda l,g=False: [k_res*l,k_res][g],
        geom_res_fc_dualB=None, geom_dualA_pred=geom_pred)

# check we do have ch3oh when lambda = 0.0
mf0 = rhf(molA)
e0 = mf0.kernel()
mf0grad = mf0.nuc_grad_method()
gradR0 = mf0grad.kernel()
mf = mfgen(0.0)
e = mf.kernel()
mfgrad = mf.nuc_grad_method()
gradR = mfgrad.kernel()
assert abs(e0 - e) < 1e-4
assert np.max(np.abs(gradR0 - gradR[:-1])) < 1e-4                                         # gradR0 only has the gradient of molA atoms while gradR has the additional molB F
assert np.max(np.abs(gradR[-1])) < 1e-4                                                   # check F has no force in the "ch3oh" state

# check we do have ch3f when lambda = 1.0
mf1 = rhf(molB)
e1 = mf1.kernel()
mf1grad = mf1.nuc_grad_method()
gradR1 = mf1grad.kernel()
mf = mfgen(1.0)
e = mf.kernel() - mf.mol.energy_geom_res()                                                # subtract the geometric restraint energy due to the dummy H
mfgrad = mf.nuc_grad_method()
gradR = mfgrad.kernel() - mf.mol.grad_geom_res()
assert abs(e1 - e) < 1e-4
assert np.max(np.abs(gradR1 - gradR[[0,1,2,3,4]])) < 1e-4
assert np.max(np.abs(gradR[5])) < 1e-4                                                    # check H has no force in the "ch3f" state
assert np.max(np.abs(gradR[6])) < 1e-4                                                    # check dummy F has no force

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
