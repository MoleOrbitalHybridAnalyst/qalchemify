def merged_scf_generator(SCF, molA, molB, singleA, singleB, dualA, dualB,
            has_grad=False,
            fsw_nelectron=None, fsw_spin=None, sigma=None,
            fsw_ham_single=None, fsw_ham_dualA=None, fsw_ham_dualB=None,
            vorb_molA=None, vorb_molB=None,
            geom_res_fc_dualA=None, geom_res_fc_dualB=None,
            geom_dualA_pred=None, geom_dualB_pred=None):
    r''' Build a merged mean-field object with interpolated Hamiltonian.

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
        a function that returns a merged mf given lambda
    '''
    from qalchemify.gto.mole import merged_mol_generator
    from pyscf.scf import uhf, ghf
    import numpy as np
    import types

    molgen = merged_mol_generator(molA, molB, singleA, singleB,
                dualA, dualB,
                has_grad=has_grad,
                fsw_nelectron=fsw_nelectron, fsw_spin=fsw_spin,
                fsw_ham_single=fsw_ham_single, fsw_ham_dualA=fsw_ham_dualA, fsw_ham_dualB=fsw_ham_dualB,
                vorb_molA=vorb_molA, vorb_molB=vorb_molB,
                geom_res_fc_dualA=geom_res_fc_dualA, geom_res_fc_dualB=geom_res_fc_dualB,
                geom_dualA_pred=geom_dualA_pred, geom_dualB_pred=geom_dualB_pred)

    def generator(l):
        mol = molgen(l)
        mf = SCF(mol)

        def get_hcore(mf, *args, **kwargs):
            h0 = mf.__class__.get_hcore(mf, *args, **kwargs)
            vorb = mf.mol.get_vorb()
            return h0 + vorb
        def get_hcore_lgrad(mf):
            return mf.mol.get_vorb_lgrad() + mf.mol.get_hcore_lgrad()

        # smearing
        do_smearing = (fsw_nelectron is not None) or (fsw_spin is not None)
        if do_smearing:
            assert sigma is not None
        is_uhf = issubclass(mf.__class__, uhf.UHF)
        is_ghf = issubclass(mf.__class__, ghf.GHF)
        is_rhf = (not is_uhf) and (not is_ghf)
        if do_smearing:
            mf.sigma = sigma(l)
        else:
            mf.sigma = None
        mf.mu = None
        if is_ghf and do_smearing:
            raise NotImplementedError()
        def fermi_occ(mu, mo_energy, sigma):
            occ = np.zeros_like(mo_energy)
            de = (mo_energy - mu) / sigma
            occ[de<40] = 1 / (np.exp(de[de<40])+1)
            return occ
        def get_occ(mf, mo_energy=None, mo_coeff=None):
            if mo_energy is None: mo_energy = mf.mo_energy
            if mo_coeff is None: mo_coeff = mf.mo_coeff
            na, nb = mol.nelec
            if is_uhf:
                moe_a, moe_b = mo_energy
                def nelec_a(mu):
                    return sum(fermi_occ(mu, moe_a, mf.sigma))
                def nelec_b(mu):
                    return sum(fermi_occ(mu, moe_b, mf.sigma))
                from scipy.optimize import root
                sol = root(lambda mu: nelec_a(mu) - na, moe_a[max(0,int(na)-1)], tol=1e-12)
                mu_a = sol['x'][0]
                sol = root(lambda mu: nelec_b(mu) - nb, moe_b[max(0,int(nb)-1)], tol=1e-12)
                mu_b = sol['x'][0]
                mf.mu = mu_a, mu_b
                mo_occ = fermi_occ(mu_a, moe_a, mf.sigma), fermi_occ(mu_b, moe_b, mf.sigma)
                occ_a, occ_b = mo_occ
            else:
                assert abs(na - nb) < 1e-5
                moe = mo_energy
                def nelec(mu):
                    return 2 * sum(fermi_occ(mu, moe, mf.sigma))
                from scipy.optimize import root
                sol = root(lambda mu: nelec(mu) - na - nb, moe[max(0,int(na)-1)], tol=1e-12)
                mf.mu = sol['x'][0]
                mo_occ = 2 * fermi_occ(mf.mu, moe, mf.sigma)
                occ_a, occ_b = mo_occ / 2, mo_occ / 2

            f = occ_a[(occ_a > 0) & (occ_a < 1)]
            mf.entropy  = -np.dot(f, np.log(f)) - np.dot(1-f, np.log(1-f))
            f = occ_b[(occ_b > 0) & (occ_b < 1)]
            mf.entropy += -np.dot(f, np.log(f)) - np.dot(1-f, np.log(1-f))
            return mo_occ

        def energy_tot(mf, *args, **kwargs):
            e = mf.__class__.energy_tot(mf, *args, **kwargs)
            e += mf.mol.energy_geom_res()
            if do_smearing:
                e -= mf.sigma * mf.entropy
            return e

        def nuc_grad_method(mf, *args, **kwargs):
            mfgrad = mf.__class__.nuc_grad_method(mf, *args, **kwargs)

            def grad_nuc(mf, mol=None, atmlst=None):
                if atmlst is not None: raise NotImplementedError()
                if mol is None:
                    mol = mf.mol
                gs = np.zeros((mol.natm,3))
                for j in range(mol.natm):
                    q2 = mol.atom_charge(j)
                    r2 = mol.atom_coord(j)
                    for i in range(mol.natm):
                        if i != j:
                            q1 = mol.atom_charge(i)
                            if q1 * q2 == 0.0:
                                continue
                            r1 = mol.atom_coord(i)
                            r = np.sqrt(np.dot(r1-r2,r1-r2))
                            gs[j] -= q1 * q2 * (r2-r1) / r**3
                return gs

            def kernel(mf, *args, **kwargs):
                gtot = mf.__class__.kernel(mf, *args, **kwargs)
                gtot += mf.mol.grad_geom_res()
                gtot[mf.mol.changeA] += gtot[mf.mol.changeB]
                gtot[mf.mol.changeB] = 0.0
                return gtot

            mfgrad.grad_nuc = types.MethodType(grad_nuc, mfgrad)
            mfgrad.kernel = types.MethodType(kernel, mfgrad)
            return mfgrad

        def energy_tot_lgrad(mf):
            dm = mf.make_rdm1()
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

        mf.get_hcore = types.MethodType(get_hcore, mf)
        mf.get_hcore_lgrad = types.MethodType(get_hcore_lgrad, mf)
        if do_smearing:
            mf.get_occ = types.MethodType(get_occ, mf)
#        mf.energy_nuc = types.MethodTypes(energy_nuc, mf)
        mf.energy_tot = types.MethodType(energy_tot, mf)
        mf.nuc_grad_method = types.MethodType(nuc_grad_method, mf)
        mf.energy_tot_lgrad = types.MethodType(energy_tot_lgrad, mf)

        mf._keys = mf._keys.union(
            ['get_hcore_lgrad', 'sigma', 'mu', 'entropy',
             'energy_tot_lgrad']
        )

        return mf

    return generator
