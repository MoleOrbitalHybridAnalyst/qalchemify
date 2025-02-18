import numpy as np
from pyscf import gto
from gpu4pyscf import scf
from gpu4qalchemify import merged_scf_generator

molA = gto.Mole()
molA.atom = \
     '''
C           -0.89307        0.58387       -0.11103
C            0.64680        0.58527       -0.13528
C           -1.41781       -0.71887        0.52089
C           -1.41781        1.80558        0.66605
H           -1.06978        1.77174        1.69389
H           -2.50260        1.82306        0.67521
H           -1.06706        2.72984        0.21897
H           -1.06705       -1.58578       -0.02918
H           -2.50260       -0.73728        0.52799
H           -1.06977       -0.80306        1.54583
H            1.02691       -0.26311       -0.69468
H            1.03602        0.52709        0.87658
H            1.02691        1.49217       -0.59375
H           -1.25497        0.64263       -1.13293 '''
molA.basis = 'sto-3g'
molA.build()

molB = gto.Mole()
molB.atom = \
     '''
C           -0.89307        0.58387       -0.11103
C            0.64680        0.58527       -0.13528
C           -1.41781       -0.71887        0.52089
C           -1.41781        1.80558        0.66605
H           -1.06978        1.77174        1.69389
H           -2.50260        1.82306        0.67521
H           -1.06706        2.72984        0.21897
H           -1.06705       -1.58578       -0.02918
H           -2.50260       -0.73728        0.52799
H           -1.06977       -0.80306        1.54583
H            1.02691       -0.26311       -0.69468
H            1.03602        0.52709        0.87658
H            1.02691        1.49217       -0.59375 '''
molB.basis = 'sto-3g'
molB.spin = 1
molB.build()

v_orb = 500   # max potential on ghost orbitals, in hartree
k_res = 0.01  # in hartree/bohr^2

def sigma(l, return_grad=False):
    # electronic temperature, in hartree
    if return_grad:
        return -0.16 * (l - 0.5)
    else:
        return -0.08 * (l - 0.5)**2 + 0.04
def f1(l, return_grad=False):
    # f1 controls potential on molA changing atom H orbitals
    if return_grad:
        return v_orb * 20 * np.exp(-20 * (1-l))
    else:
        return v_orb * np.exp(-20 * (1-l))
def g0(l, return_grad=False):
    # g0 interpolates Hamiltonian
    if return_grad:
        return -1
    else:
        return 1 - l
def geom_pred(x, return_grad=False):
    # gives rougly the dummy H position given the shared atom coords (i.e. butyl positions)
    x12, y12, z12 = x[2] - x[1]
    x02, y02, z02 = x[2] - x[0]
    x01, y01, z01 = x[1] - x[0]
    h_pos = x[0]
    h_pos[0] += 1/4 * (y01 * z02 - z01 * y02)
    h_pos[1] += 1/4 * (z01 * x02 - x01 * z02)
    h_pos[2] += 1/4 * (x01 * y02 - y01 * x02)
    if not return_grad:
        return h_pos
    else:
        pass
    g = np.zeros((1,3,13,3))
    drX_dr0 = np.array([
            [4, z12,  -y12],
            [-z12, 4,  x12],
            [y12, -x12,  4]]) / 4
    drX_dr1 = np.array([
            [0.00, -z02,  y02],
            [z02,  0.00, -x02],
            [-y02,  x02,  0.00]]) / 4
    drX_dr2 = np.array([
            [0.00, z01,  -y01],
            [-z01, 0.00,  x01],
            [y01, -x01,  0.00]]) / 4
    g[:,:,0,:] = drX_dr0.T
    g[:,:,1,:] = drX_dr1.T
    g[:,:,2,:] = drX_dr2.T
    return g

def uhf(mol, use1e=True):
    mf = scf.UHF(mol)
    if use1e:
        mf.init_guess = '1e'
    return mf
mfgen = merged_scf_generator(uhf, molA, molB,
        list(range(13)), list(range(13)),
        [13], [],
        has_grad=True,
        fsw_nelectron=g0, fsw_spin=g0, sigma=sigma,
        fsw_ham_single=None, fsw_ham_dualA=g0, fsw_ham_dualB=None,
        vorb_molA=f1,
        vorb_molB=None,
        geom_res_fc_dualA=lambda l,g=False: [k_res*(1-g0(l)),-k_res*g0(l,True)][g],
        geom_res_fc_dualB=None,
        geom_dualA_pred=geom_pred, geom_dualB_pred=None)

# check we do have butane when lambda = 0.0
mf0 = uhf(molA, use1e=False)
e0 = mf0.kernel()
mf0grad = mf0.nuc_grad_method()
gradR0 = mf0grad.kernel()
mf = mfgen(0.0)
e = mf.kernel()
mfgrad = mf.nuc_grad_method()
gradR = mfgrad.kernel()
assert abs(e0 - e) < 1e-4
assert np.max(np.abs(gradR0 - gradR)) < 1e-4

# check we do have butyl when lambda = 1.0
mf1 = uhf(molB, use1e=False)
e1 = mf1.kernel()
mf1grad = mf1.nuc_grad_method()
gradR1 = mf1grad.kernel()
mf = mfgen(1.0)
e = mf.kernel() - mf.mol.energy_geom_res()                                                # subtract the geometric restraint energy due to the dummy H
mfgrad = mf.nuc_grad_method()
gradR = mfgrad.kernel() - mf.mol.grad_geom_res()
assert abs(e1 - e) < 1e-4
assert np.max(np.abs(gradR1 - gradR[range(13)])) < 1e-4
assert np.max(np.abs(gradR[13])) < 1e-4                                                   # check dummy H has no force in the butyl state

# check lambda gradient using finite difference
def check_lambda_gradient(l0):
    dEdl = mfgen(l0).run().energy_tot_lgrad()
    dEdlnum = (mfgen(l0+1e-5).kernel() - mfgen(l0-1e-5).kernel()) / 2e-5
    assert abs(dEdl - dEdlnum) < 1e-5
check_lambda_gradient(0.999)
check_lambda_gradient(0.05)
check_lambda_gradient(0.5)
check_lambda_gradient(0.95)
check_lambda_gradient(0.001)
