from argparse import ArgumentParser, Namespace
import os
import CANalyzer.mo
import CANalyzer.projection
import CANalyzer.natorb

parser = ArgumentParser()
parser.add_argument('jobtype', help='Type of job to perform', type=str)
"""
    jobtype: MOAnalyzer - provides AO contribution to MOs partitioned by angular momentum and atom (groups of atoms)
             Projection - Provides projection of MOs in fchk onto MOs in fchk2 (Only for Gaussian)
             NatOrb - Computes natural orbitals and stores it into a new fchk (Only for Gaussian)
"""
parser.add_argument('--log', help='Full directory path to log file or out file', type=str)
parser.add_argument('--fchk', help='Full directory path to fchk or bin file', type=str)
parser.add_argument('--filename', help='Custom filename to save created files', type=str)
parser.add_argument('--log2', help='Full directory path to log file or out file', type=str)
parser.add_argument('--fchk2', help='Full directory path to fchk file', type=str)
parser.add_argument('--ml', help='True if MOAnalyzer separates based on ml number as well', type=bool)
parser.add_argument("--groups", help='Dictionary of custom atom groupings or range of states for NatOrb', type=str)
"""
    groups: MOAnalyzer - Make groups of atoms you want to group together in output. If you have atoms (listed in order)
                         C, F, Cl, Br, C, H, H, C, O, O, H
                         and you want to group the Halogens, Carbons, and Others.
                         --groups="{'Halogens':'[2:4]', 'Carbons':'[1:1,5:5,8:8]', 'Others':'[6:7,9:11]'}"
                         
            NatOrb - CI state which natural orbitals are generated from
"""
parser.add_argument("--displaywidth", help='Display width of output file before skipping line', type=int)
args: Namespace = parser.parse_args()

# setting defaults
directory = os.getcwd()
filename = directory + "/"
if args.filename:
    filename += args.filename

if args.displaywidth:
    displaywidth = args.displaywidth
else:
    displaywidth = 100000

# running jobs
if args.jobtype == "MOAnalyzer":
    if not args.filename:
        filename += 'orbitals.txt'
    if not args.ml:
        args.ml = False
    MOAnalyzer = CANalyzer.mo.MO(args.log, args.fchk, filename, args.groups, displaywidth, args.ml)
    MOAnalyzer.start()
    MOAnalyzer.mulliken_analysis()
    MOAnalyzer.print_mulliken()
elif args.jobtype == 'Projection':
    if not args.filename:
        filename += 'projections.txt'
    Projection = CANalyzer.projection.Projection(args.log, args.fchk, args.fchk2, filename, displaywidth)
    Projection.start()
    Projection.project()
    Projection.print_project()
elif args.jobtype == 'NatOrb':
    if not args.filename:
        filename += 'natorb'
    NatOrb = CANalyzer.natorb.NaturalOrbitals(args.log, args.fchk, filename, args.groups, displaywidth)
    NatOrb.start()
    NatOrb.compute_natorb()
elif args.jobtype == 'MoveMO':
    if not args.fchk2:
        raise Exception("Set --fchk2 to target fchk")
    from CANalyzer.load import Load
    from CANalyzer.utilities import write_fchk
    Data1 = Load(args.log, args.fchk, filename, args.groups, displaywidth)
    Data1.start()
    matsize = Data1.nbasis * Data1.nbsuse * Data1.ncomp * Data1.ncomp
    moa, mob = Data1.read_mo()
    write_fchk("Alpha MO coefficients", Data1.nri, moa, matsize, Data1.fchkfile, args.fchk2)
    if Data1.xhf == "UHF":
        write_fchk("Beta MO coefficients", Data1.nri, mob, matsize, Data1.fchkfile, args.fchk2)
else:
    print("Invalid jobtype")






