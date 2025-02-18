def merged_mol_generator(molA, molB, singleA, singleB, dualA, dualB,
        has_grad=False,
        fsw_nelectron=None, fsw_spin=None,
        fsw_ham_single=None, fsw_ham_dualA=None, fsw_ham_dualB=None,
        vorb_molA=None, vorb_molB=None,
        geom_res_fc_dualA=None, geom_res_fc_dualB=None,
        geom_dualA_pred=None, geom_dualB_pred=None):
    r''' Merge two mol into an interpolated mol at the Hamiltonian level.
         If molA and molB have inconsistent attributes, the ones of molA will be used.

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
            if True, switching functions, vorb, geom_res_fc expected to return the lambda gradient
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

    Returns:
        a function that returns a merged mol given lambda
    '''
    import cupy as cp
    import numpy as np
    from pyscf import gto
    import types

    molA = molA.copy()
    molB = molB.copy()
    molA.build()
    molB.build()

    # identify the changing atoms within singleA and singleB
    changeA = list()   # atom indices in molA
    changeB = list()   # atom indices in molB
    if fsw_ham_single is not None:
        for i, j in zip(singleA, singleB):
            if molA.atom_charge(i) != molB.atom_charge(j) or \
               molA.atom_nelec_core(i) != molB.atom_nelec_core(j):
                changeA.append(i)
                changeB.append(j)

    # sanity checks
    assert len(singleA) == len(singleB)
    assert len(singleA) + len(dualA) == molA.natm
    assert len(singleB) + len(dualB) == molB.natm
    check_fsw = lambda fsw: (abs(fsw(0.0) - 1.0) < 1e-5) and (abs(fsw(1.0) - 0.0) < 1e-5)
    check_resA = lambda res: (abs(res(0.0)) < 1e-5) and (res(1.0) > 0)
    check_resB = lambda res: (abs(res(1.0)) < 1e-5) and (res(0.0) > 0)
    if fsw_nelectron is not None:
        assert check_fsw(fsw_nelectron)
    else:
        assert molA.tot_electrons() == molB.tot_electrons()
    if fsw_spin is not None:
        assert check_fsw(fsw_spin)
    else:
        assert molA.spin == molB.spin
    if fsw_ham_single is not None:
        assert check_fsw(fsw_ham_single)
    else:
        assert (molA.atom_charges()[singleA] == molB.atom_charges()[singleB]).all()
    if fsw_ham_dualA is not None:
        assert check_fsw(fsw_ham_dualA)
    else:
        assert len(dualA) == 0
    if fsw_ham_dualB is not None:
        assert check_fsw(fsw_ham_dualB)
    else:
        assert len(dualB) == 0
    if vorb_molA is not None:
        assert check_resA(vorb_molA)
    else:
        assert len(changeA) == 0 and len(dualA) == 0
    if vorb_molB is not None:
        assert check_resB(vorb_molB)
    else:
        assert len(changeB) == 0 and len(dualB) == 0
    if geom_res_fc_dualA is not None:
        assert geom_dualA_pred is not None
        assert check_resA(geom_res_fc_dualA)
    else:
        assert len(dualA) == 0
    if geom_res_fc_dualB is not None:
        assert geom_dualB_pred is not None
        assert check_resB(geom_res_fc_dualB)
    else:
        assert len(dualB) == 0

    # check if changing atoms have ECP
    for i in changeA + dualA:
        if molA.atom_nelec_core(i) != 0:
            raise NotImplementedError()
    for i in changeB + dualB:
        if molB.atom_nelec_core(i) != 0:
            raise NotImplementedError()

    def generator(l):
        mol = molA.copy()
        atom = mol._atom
        for i, j in zip(changeA, changeB):
            atom.append(("X-" + molB._atom[j][0], molA._atom[i][1]))
        for j in dualB:
            # TODO add ECP for ghost atoms
            atom.append(("X-" + molB._atom[j][0], molB._atom[j][1]))
        mol.atom = atom
        mol.charge = 0
        mol._nelectron = 0
        mol.spin = 0
        mol.build(unit='Bohr')
        mol.charge = molA.charge
        mol._nelectron = molA.tot_electrons()
        mol.spin = molA.spin

        # atom indices in merged mol:
        mchangeA = list(changeA)
        mchangeB = list(range(molA.natm, molA.natm+len(changeB)))
        mdualA   = list(dualA)
        mdualB   = list(range(molA.natm+len(changeB), molA.natm+len(changeB)+len(dualB)))

        if fsw_nelectron is not None:
            mol._nelectron  = molA.tot_electrons() * fsw_nelectron(l)
            mol._nelectron += molB.tot_electrons() * (1 - fsw_nelectron(l))
            mol.tot_electrons = lambda *args: mol._nelectron

        if fsw_spin is not None:
            mol.spin  = molA.spin * fsw_spin(l)
            mol.spin += molB.spin * (1 - fsw_spin(l))

        # interpolate the nuclear charges
        # TODO interpolate ECP
        nuc_chargesA = molA.atom_charges()
        nuc_chargesB = molB.atom_charges()
        nuc_charges = np.array(mol.atom_charges(), float)
        for i, j, i_ in zip(changeA, changeB, mchangeA):
            nuc_charges[i_]  = nuc_chargesA[i] * fsw_ham_single(l)
            nuc_charges[i_] += nuc_chargesB[j] * (1 - fsw_ham_single(l))
        for i, i_ in zip(dualA, mdualA):
            nuc_charges[i_]  = nuc_chargesA[i] * fsw_ham_dualA(l)
        for i, i_ in zip(dualB, mdualB):
            nuc_charges[i_]  = nuc_chargesB[i] * (1 - fsw_ham_dualB(l))
        # makes nuc_charges effective
        offset = mol._env.size
        mol._atm[:, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
        mol._env = np.append(mol._env, nuc_charges)
        mol._atm[:, gto.PTR_FRAC_CHARGE] = offset + np.arange(mol.natm)

        mol.charge = sum(mol.atom_charges()) + \
                     sum([mol.atom_nelec_core(i) for i in range(mol.natm)]) - \
                     mol._nelectron

        if fsw_nelectron is None and fsw_spin is None:
            # not do smearing then nelectron and spin should be integers
            assert abs(mol._nelectron - int(mol._nelectron)) < 1e-12
            assert abs(mol.spin - int(mol.spin)) < 1e-12
            assert abs(mol.charge - int(mol.charge)) < 1e-12
            mol._nelectron = int(mol._nelectron)
            mol.spin = int(mol.spin)
            mol.charge = int(mol.charge)
        else:
            # do smearing. overwrite the original nelec that assumes integers
            nelec = (
                (mol._nelectron + mol.spin) / 2,
                (mol._nelectron - mol.spin) / 2 )
            mol.__class__ = type("MergedMole", (mol.__class__,), {
                "nelec": nelec
            })

        def get_nelec_lgrad(mol):
            if fsw_nelectron is None and fsw_spin is None:
                return (0, 0)
            else:
                if fsw_nelectron is not None:
                    nelectron_lgrad = (molA.tot_electrons() - molB.tot_electrons()) \
                        * fsw_nelectron(l, True)
                else:
                    nelectron_lgrad = 0
                if fsw_spin is not None:
                    spin_lgrad = (molA.spin - molB.spin) * fsw_spin(l, True)
                else:
                    spin_lgrad = 0
                return (
                    (nelectron_lgrad + spin_lgrad) / 2,
                    (nelectron_lgrad - spin_lgrad) / 2 )

        def get_vorb(mol):
            aoslices = mol.aoslice_by_atom()
            vorb = cp.zeros([mol.nao]*2)
            for i in mchangeA + mdualA:
                p0, p1 = aoslices[i][-2:]
                cp.fill_diagonal(vorb[p0:p1,p0:p1], vorb_molA(l))
            for i in mchangeB + mdualB:
                p0, p1 = aoslices[i][-2:]
                cp.fill_diagonal(vorb[p0:p1,p0:p1], vorb_molB(l))
            return vorb

        def get_vorb_lgrad(mol):
            assert has_grad
            aoslices = mol.aoslice_by_atom()
            vorb = np.zeros([mol.nao]*2)
            for i in mchangeA + mdualA:
                p0, p1 = aoslices[i][-2:]
                np.fill_diagonal(vorb[p0:p1,p0:p1], vorb_molA(l, True))
            for i in mchangeB + mdualB:
                p0, p1 = aoslices[i][-2:]
                np.fill_diagonal(vorb[p0:p1,p0:p1], vorb_molB(l, True))
            return vorb

        def get_hcore_lgrad(mol):
            assert has_grad
            hcore = np.zeros([mol.nao]*2)
            for i, j, i_ in zip(changeA, changeB, mchangeA):
                with mol.with_rinv_origin(mol.atom_coord(i_)):
                    hcore -= mol.intor('int1e_rinv') * \
                        (nuc_chargesA[i] - nuc_chargesB[j]) * fsw_ham_single(l, True)
            for i, i_ in zip(dualA, mdualA):
                with mol.with_rinv_origin(mol.atom_coord(i_)):
                    hcore -= mol.intor('int1e_rinv') * nuc_chargesA[i] * \
                        fsw_ham_dualA(l, True)
            for i, i_ in zip(dualB, mdualB):
                with mol.with_rinv_origin(mol.atom_coord(i_)):
                    hcore += mol.intor('int1e_rinv') * nuc_chargesB[i] * \
                        fsw_ham_dualB(l, True)
            return hcore

        def energy_geom_res(mol):
            egr = 0.0
            if geom_dualA_pred is not None:
                x = geom_dualA_pred(mol.atom_coords()[singleA])
                egr += geom_res_fc_dualA(l) * \
                    cp.sum( (x - mol.atom_coords()[mdualA])**2 )
            if geom_dualB_pred is not None:
                x = geom_dualB_pred(mol.atom_coords()[singleA])
                egr += geom_res_fc_dualB(l) * \
                    cp.sum( (x - mol.atom_coords()[mdualB])**2 )
            return egr

        def grad_geom_res(mol):
            grad = np.zeros((mol.natm,3))

            if geom_dualA_pred is not None:
                x = geom_dualA_pred(mol.atom_coords()[singleA])
                dxdR = geom_dualA_pred(mol.atom_coords()[singleA], True)
                grad[mdualA] = 2 * geom_res_fc_dualA(l) * \
                    (mol.atom_coords()[mdualA] - x)
                grad[singleA] -= \
                    np.einsum('ix,ixjy->jy', grad[mdualA], dxdR)

            if geom_dualB_pred is not None:
                x = geom_dualB_pred(mol.atom_coords()[singleA])
                dxdR = geom_dualB_pred(mol.atom_coords()[singleA], True)
                grad[mdualB] = 2 * geom_res_fc_dualB(l) * \
                    (mol.atom_coords()[mdualB] - x)
                grad[singleA] -= \
                    np.einsum('ix,ixjy->jy', grad[mdualB], dxdR)

            return grad

        def energy_geom_res_lgrad(mol):
            egr = 0.0
            if geom_dualA_pred is not None:
                x = geom_dualA_pred(mol.atom_coords()[singleA])
                egr += geom_res_fc_dualA(l, True) * \
                    cp.sum( (x - mol.atom_coords()[mdualA])**2 )
            if geom_dualB_pred is not None:
                x = geom_dualB_pred(mol.atom_coords()[singleA])
                egr += geom_res_fc_dualB(l, True) * \
                    cp.sum( (x - mol.atom_coords()[mdualB])**2 )
            return egr


        def energy_nuc_lgrad(mol):
            rr = mol.atom_coords()[:,None,:] - mol.atom_coords()[None]
            rr = np.linalg.norm(rr, axis=-1)
            rr[np.diag_indices_from(rr)] = 1e200
            for i, j in zip(mchangeA, mchangeB):
                rr[i, j] = 1e200
                rr[j, i] = 1e200
            dq = np.zeros(mol.natm)
            for i, j, i_ in zip(changeA, changeB, mchangeA):
                dq[i_]  = (nuc_chargesA[i] - nuc_chargesB[j]) * fsw_ham_single(l, True)
            for i, i_ in zip(dualA, mdualA):
                dq[i_]  = nuc_chargesA[i] * fsw_ham_dualA(l, True)
            for i, i_ in zip(dualB, mdualB):
                dq[i_]  = -nuc_chargesB[i] * fsw_ham_dualB(l, True)
            eg = np.einsum('i,ij,j->', dq, 1./rr, mol.atom_charges())
            return eg

        mol.get_vorb = types.MethodType(get_vorb, mol)
        mol.get_vorb_lgrad = types.MethodType(get_vorb_lgrad, mol)
        mol.get_hcore_lgrad = types.MethodType(get_hcore_lgrad, mol)
        mol.energy_geom_res = types.MethodType(energy_geom_res, mol)
        mol.grad_geom_res = types.MethodType(grad_geom_res, mol)
        mol.energy_geom_res_lgrad = types.MethodType(energy_geom_res_lgrad, mol)
        mol.energy_nuc_lgrad = types.MethodType(energy_nuc_lgrad, mol)
        mol.get_nelec_lgrad = types.MethodType(get_nelec_lgrad, mol)
        mol.natmA = molA.natm
        mol.natmB = molB.natm
        mol.changeA = mchangeA
        mol.changeB = mchangeB
        mol.dualA = mdualA
        mol.dualB = mdualB

        mol._keys = mol._keys.union(
            ['changeA', 'changeB', 'dualA', 'dualB', 'natmA', 'natmB',
             'get_vorb', 'get_vorb_lgrad', 'get_hcore_lgrad', 'get_nelec_lgrad',
             'energy_geom_res', 'grad_geom_res',
             'energy_geom_res_lgrad', 'energy_nuc_lgrad'])

        return mol

    return generator