import numpy as np
import pandas as pd
from CANalyzer.load import Load
import CANalyzer.utilities as util
import sys

class MO(Load):
    def __init__(self, logfile, fchkfile, filename, groups):
        Load.__init__(self, logfile, fchkfile, filename, groups)
        self.moalpha = None
        self.mobeta = None
        self.overlap = None
        self.alpha_pop = None
        self.beta_pop = None
        self.sorted_alphapop = None
        self.sorted_betapop = None
        self.reduced_groups = None

    def mulliken_analysis(self):
        self.overlap = self.read_overlap()
        self.moalpha, self.mobeta = self.read_mo()
        cs = np.matmul(self.overlap, self.moalpha)
        if self.nri == 1:
            pop = np.multiply(self.moalpha, cs)
        else:
            conj_moalpha = np.conj(np.copy(self.moalpha))
            pop = np.real(np.multiply(conj_moalpha, cs))

        if self.xhf in ['GHF', 'GCAS', 'UHF']:
            self.alpha_pop = pop
            csbeta = np.matmul(self.overlap, self.mobeta)
            if self.nri == 1:
                self.beta_pop = np.multiply(self.mobeta, csbeta)
            elif self.nri == 2:
                conj_mobeta = np.conj(np.copy(self.mobeta))
                self.beta_pop = np.real(np.multiply(conj_mobeta, csbeta))
        else:
            self.alpha_pop = pop

        if not self.groups:
            self.reduced_groups = []
            [self.reduced_groups.append(x[1:]) for x in self.subshell if x[1:] not in self.reduced_groups]
            self.sorted_alphapop = np.zeros((len(self.reduced_groups), self.nbsuse*self.ncomp))
            self.sorted_betapop = np.zeros((len(self.reduced_groups), self.nbsuse*self.ncomp))
            for j in range(self.nbsuse*self.ncomp):
                for i in range(self.nbasis):
                    atomshellpair = self.subshell[i][1:]
                    index = self.reduced_groups.index(atomshellpair)
                    self.sorted_alphapop[index, j] += self.alpha_pop[i, j]
                    if self.xhf in ['UHF', 'GHF', 'GCAS']:
                        self.sorted_betapop[index, j] += self.beta_pop[i, j]
        else:
            self.reduced_groups = [(g, l) for g in self.groupnames for l in range(self.maxL + 1)]
            self.sorted_alphapop = np.zeros((len(self.reduced_groups), self.nbsuse*self.ncomp))
            self.sorted_betapop = np.zeros((len(self.reduced_groups), self.nbsuse*self.ncomp))
            for j in range(self.nbsuse*self.ncomp):
                for i in range(self.nbasis):
                    atomshellpair = self.subshell[i]
                    atom_num = atomshellpair[0]
                    l = util.OAM[atomshellpair[2]]
                    group = self.identify_group(atom_num)
                    index = self.reduced_groups.index((group, l))
                    self.sorted_alphapop[index, j] += self.alpha_pop[i, j]
                    if self.xhf in ['UHF', 'GHF', 'GCAS']:
                        self.sorted_betapop[index, j] += self.beta_pop[i, j]

    def identify_group(self, atom_number):
        for name in self.groupnames:
            grouprange = self.groups[name]
            for r in grouprange:
                start = r[0]
                end = r[1] + 1
                if atom_number in range(start, end):
                    group = name
                    break
        return group

    def print_mulliken(self):
        if self.xhf in ['GHF', 'GCAS']:
            results_alpha = dict(zip(self.reduced_groups, (self.sorted_alphapop + self.sorted_betapop).round(3)))
            remark_alpha = f"\n{self.xhf} Orbitals\n"
            alpha_spin = [np.sum(self.alpha_pop[:,i]) for i in range(self.nbsuse*self.ncomp)]
            beta_spin = [np.sum(self.beta_pop[:, i]) for i in range(self.nbsuse*self.ncomp)]
            self.spin = dict(zip(['Alpha', 'Beta'], [alpha_spin, beta_spin]))
        elif self.xhf in ['RHF', 'ROHF', 'RCAS']:
            results_alpha = dict(zip(self.reduced_groups, self.sorted_alphapop.round(3)))
            remark_alpha = f"\n{self.xhf} Orbitals\n"
        else:
            results_alpha = dict(zip(self.reduced_groups, self.sorted_alphapop.round(3)))
            remark_alpha = f"\n{self.xhf} Alpha Orbitals\n"
            results_beta = dict(zip(self.reduced_groups, self.sorted_betapop.round(3)))
            remark_beta = f"\n{self.xhf} Beta Orbitals\n"

        with open(self.filename, 'w') as sys.stdout, pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(f'Number of Alpha Electrons: {self.nae}   Beta Electrons: {self.nbe}')
            print(f'Number of AOs: {self.nbasis*self.ncomp}    MOs: {self.nbsuse*self.ncomp}')
            results_alpha_df = pd.DataFrame(results_alpha)
            print(remark_alpha)
            print(results_alpha_df)
            if self.xhf in ['UHF']:
                results_beta_df = pd.DataFrame(results_beta)
                print(remark_beta)
                print(results_beta_df)
            elif self.xhf in ['GHF', 'GCAS']:
                spin_df = pd.DataFrame(self.spin)
                print('\nSpin Contribution\n')
                print(spin_df)



















