def merged_posthf_generator(SCF, PostHF, molA, molB, singleA, singleB, dualA, dualB,
            has_grad=False,
            fsw_nelectron=None, fsw_spin=None, sigma=None,
            fsw_ham_single=None, fsw_ham_dualA=None, fsw_ham_dualB=None,
            vorb_molA=None, vorb_molB=None,
            geom_res_fc_dualA=None, geom_res_fc_dualB=None,
            geom_dualA_pred=None, geom_dualB_pred=None):
    r''' Build a merged post Hartree-Fock object with interpolated Hamiltonian.

    Args:
        molA:
            mol when lambda = 0.0
        molB:
            mol when lambda = 1.0
        singleA (List):
            coordinate shared atom indices in molA
        singleB (List):
            coordinate shared atom indices in molB
        dualA (List):
            unique atom indices in molA
        dualB (List):
            unique atom indices in molB
        has_grad (bool):
            if True, switching functions, vorb, geom_res_fc, sigma expected to return the lambda gradient
             when taking an extra argument return_grad, while geom_pred expected to return the nuc gradient
        fsw_nelectron (Callable):
            switching function that controls nelectron given lambda.
            expect fsw(0.0), fsw(1.0) = 1.0, 0.0
            nelectron(l) = nelectronA * fsw(l) + nelectronB * (1 - fsw(l))
        fsw_spin (Callable):
            switching function that controls spin given lambda
            expect fsw(0.0), fsw(1.0) = 1.0, 0.0
        fsw_ham_single (Callable):
            switching function that controls alchemical transition between coordinate shared atoms.
            expect fsw(0.0), fsw(1.0) = 1.0, 0.0
            for those atoms, H = HA * fsw(l) + HB * (1 - fsw(l))
        fsw_ham_dualA (Callable):
            switching function that controls the annihilation of dualA atoms.
            expect fsw(0.0), fsw(1.0) = 1.0, 0.0
            for dualA atoms, H = HA * fsw(l)
        fsw_ham_dualB (Callable):
            switching function that controls the annihilation of dualB atoms.
            expect fsw(0.0), fsw(1.0) = 1.0, 0.0
            for dualB atoms, H = HB * (1 - fsw(l))
        vorb_molA (Callable):
            returns the potential on ghost orbitals of molA atoms given lambda
        vorb_molB (Callable):
            returns the potential on ghost orbitals of molB atoms given lambda
        geom_res_fc_dualA (Callable):
            returns the force constant for restraining dualA atoms
        geom_res_fc_dualB (Callable):
            returns the force constant for restraining dualB atoms
        geom_dualA_pred:
            predict dualA atom positions given shared coordinates
        geom_dualB_pred:
            predict dualB atom positions given shared coordinates
        sigma (Callable):
            returns the electronic temperature given lambda

    Returns:
        a function that returns a merged post hf given lambda
    '''
    from qalchemify.scf.hf import merged_scf_generator
    import numpy as np
    import types
    from pyscf.scf import uhf, ghf

    mfgen = merged_scf_generator(SCF, molA, molB, singleA, singleB, dualA, dualB,
            has_grad=has_grad,
            fsw_nelectron=fsw_nelectron, fsw_spin=fsw_spin, sigma=sigma,
            fsw_ham_single=fsw_ham_single, fsw_ham_dualA=fsw_ham_dualA, fsw_ham_dualB=fsw_ham_dualB,
            vorb_molA=vorb_molA, vorb_molB=vorb_molB,
            geom_res_fc_dualA=geom_res_fc_dualA, geom_res_fc_dualB=geom_res_fc_dualB,
            geom_dualA_pred=geom_dualA_pred, geom_dualB_pred=geom_dualB_pred)

    def generator(l):
        mf = mfgen(l)
        mcc = PostHF(mf)

        # allow mcc to find mo_occ and mo_coeff after mcc._scf is called
        mcc.__class__ = type("Merged" + mcc.__class__.__name__,
                             (mcc.__class__,),
                             {"mo_occ": property(lambda self: self._scf.mo_occ),
                              "mo_coeff": property(lambda self: self._scf.mo_coeff)})

        # smearing
        do_smearing = (fsw_nelectron is not None) or (fsw_spin is not None)
        if do_smearing:
            raise NotImplementedError('finite temperature Post HF not supported yet')
        is_uhf = issubclass(mf.__class__, uhf.UHF)
        is_ghf = issubclass(mf.__class__, ghf.GHF)
        is_rhf = (not is_uhf) and (not is_ghf)

        def nuc_grad_method(mcc, *args, **kwargs):
            mccgrad = mcc.__class__.nuc_grad_method(mcc, *args, **kwargs)

            # mcc._scf.nuc_grad_method().grad_nuc will be called in mcc.__class__.kernel()
            # but mcc._scf.nuc_grad_method().kernel will NOT be called
            # so a modified kernel is needed here but not grad_nuc

            def kernel(mcc, *args, **kwargs):
                gtot = mcc.__class__.kernel(mcc, *args, **kwargs)
                mol = mcc.base._scf.mol
                gtot += mol.grad_geom_res()
                gtot[mol.changeA] += gtot[mol.changeB]
                gtot[mol.changeB] = 0.0
                return gtot

            mccgrad.kernel = types.MethodType(kernel, mccgrad)
            return mccgrad

        def energy_tot_lgrad(mcc):
            dm = mcc.make_rdm1(relaxed=True)
            mf = mcc._scf
            h1_lgrad = mf.mol.get_hcore_lgrad() + mf.mol.get_vorb_lgrad()
            if is_rhf:
                eg = np.einsum('pq,qp->', h1_lgrad, dm)
            elif is_uhf:
                eg = np.einsum('pq,xqp->', h1_lgrad, dm)
            else:
                raise NotImplementedError()
            eg += mf.mol.energy_geom_res_lgrad()
            eg += mf.mol.energy_nuc_lgrad()
            if do_smearing:
                if is_uhf:
                    eg += np.dot(mf.mu, mf.mol.get_nelec_lgrad())
                else:
                    eg += mf.mu * sum(mf.mol.get_nelec_lgrad())
                eg -= mf.entropy * sigma(l, True)
            return eg

        mcc.nuc_grad_method = types.MethodType(nuc_grad_method, mcc)
        mcc.energy_tot_lgrad = types.MethodType(energy_tot_lgrad, mcc)
        mcc._keys = mcc._keys.union(
            ['energy_tot_lgrad']
        )


        return mcc

    return generator