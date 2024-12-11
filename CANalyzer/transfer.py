import numpy as np
from CANalyzer.load import Load
import CANalyzer.utilities as util
import os
import subprocess

class Tranfer():
    def __init__(self, from_logfile, from_fchkfile, to_logfile, to_fchkfile):
        self.fromjob = Load(from_logfile, from_fchkfile)
        self.tojob = None

        # CQ need existing bin to write into
        if self.fromjob.software == "CQ":
            if to_logfile and to_fchkfile:
                self.tojob = Load(to_logfile, to_fchkfile)
                self.tojob.start()
            else:
                print("CQ job detected but no target specified.")
                print(f"Assuming target is CQ. Copying {from_fchkfile} to make target: target.bin")
                os.system(f"cp {from_fchkfile} target.bin")
                to_logfile = from_logfile
                to_fchkfile = "target.bin"
                self.tojob = Load(to_logfile, to_fchkfile)
                self.tojob.start()

        else:
            if to_logfile and to_fchkfile:
                self.tojob = Load(to_logfile, to_fchkfile)
                self.tojob.start()
            else:
                # since fchk files are written on the spot, not edited like bin files, we don't have an object for the target
                # target variable is set to job since their data should be the same
                self.tojob = self.fromjob

        self.fromjob.start()


    def swapmo(self, pairs):
        if self.fromjob.software != self.tojob.software:
            raise Exception("SwapMO require job and target to be in the same software")

        # does not work for UHF
        mo, void = self.fromjob.read_mo(separate=False)

        # swapping pairs
        for f, t in pairs:
            holdf = np.copy(mo[:, f - 1])
            holdt = np.copy(mo[:, t - 1])
            mo[:, f - 1] = holdt
            mo[:, t - 1] = holdf

        if self.fromjob.software == "CQ":
            mo = mo.T
            self.tojob.writebin_matrix("/SCF/MO1", mo)
        else:
            matsize = self.fromjob.nbasis * self.fromjob.nbsuse * self.fromjob.ncomp * self.fromjob.ncomp
            util.write_fchk("Alpha MO coefficients", self.fromjob.nri, mo, matsize, self.fromjob.fchkfile, "target.fchk")


    def movemo(self):
        if self.fromjob.nbasis != self.tojob.nbasis or self.fromjob.nbsuse != self.tojob.nbsuse:
            raise Exception(
                "To and from jobs must have same basis. Make sure GDV job did not prune for linear dependencies.")

        if self.fromjob.nri != self.tojob.nri or self.fromjob.ncomp != self.tojob.ncomp:
            raise Exception("To and from jobs must be same component and real/complex.")

        swap_pairs = []
        n = 0
        #for n in range(self.fromjob.nbasis):
        while n < self.fromjob.nbasis:
            oam = util.OAM[self.fromjob.subshell[n][2]]
            if oam >= 2:
                # assigning positions, not swapping pairs
                # (old, new) if fromjob is CQ
                # (new, old) if fromjob is GDV
                for i in range(2*oam+1):
                    new = oam - i
                    if new < 0:
                        new = -2 * new - 1 + n
                    else:
                        new = 2 * new + n

                    swap_pairs.append((i+n, new))

            n += 2*oam + 1

        moa, mob = self.fromjob.read_mo()

        if self.fromjob.software == "CQ" and self.tojob.software == "GDV":
            oldrows = [np.copy(moa[p[0], :]) for p in swap_pairs]
            if self.fromjob.ncomp == 2:
                oldrows_b = [np.copy(mob[p[0], :]) for p in swap_pairs]

            for p in range(len(swap_pairs)):
                oldrow_index = swap_pairs[p][0]
                newrow_index = swap_pairs[p][1]
                moa[newrow_index, :] = oldrows[p]
                if self.fromjob.ncomp == 2:
                    mob[newrow_index, :] = oldrows_b[p]

            if self.fromjob.ncomp == 2:
                new_mo = np.zeros(
                    (self.fromjob.nbasis * self.fromjob.ncomp, self.fromjob.nbsuse * self.fromjob.ncomp), dtype=np.complex128)
                for i in range(self.fromjob.nbasis):
                    new_mo[2 * i + 1, :] = mob[i, :]
                    new_mo[2 * i, :] = moa[i, :]

            elif self.fromjob.ncomp == 4:
                raise Exception("4-Component NYI in GDV")

            else:
                new_mo = moa

            util.write_fchk("Alpha MO coefficients", self.fromjob.nri, new_mo,
                            self.fromjob.nbasis * self.fromjob.nbsuse * self.fromjob.ncomp * self.fromjob.ncomp,
                            self.tojob.fchkfile, "target.fchk")

        elif self.fromjob.software == "GDV" and self.tojob.software == "CQ":
            oldrows = [np.copy(moa[p[1], :]) for p in swap_pairs]
            if self.fromjob.ncomp == 2:
                oldrows_b = [np.copy(mob[p[1], :]) for p in swap_pairs]
            for row in range(len(swap_pairs)):
                newrow = swap_pairs[row][0]
                moa[newrow, :] = oldrows[row]
                if self.fromjob.ncomp == 2:
                    mob[newrow, :] = oldrows_b[row]

            if self.fromjob.ncomp == 2:
                new_mo = np.zeros(
                    (self.fromjob.nbasis * self.fromjob.ncomp, self.fromjob.nbsuse * self.fromjob.ncomp), dtype=np.complex128)
                new_mo[0:self.fromjob.nbasis, :] = moa
                new_mo[self.fromjob.nbasis:, :] = mob

            elif self.fromjob.ncomp == 4:
                raise Exception("4-Component NYI in GDV")

            else:
                new_mo = moa

            self.tojob.writebin_matrix("/SCF/MO1", new_mo.T)

        else:
            raise Exception("MOs can only be moved between jobs from different software")










