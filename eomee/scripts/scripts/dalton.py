import os, sys 
from glob import glob

from string import Template

from argparse import ArgumentParser

from sys import stderr


# Initialize command line argument parser
parser = ArgumentParser()


# Specify positional arguments and options
parser.add_argument("-r", action="store_true", default=False, help="submit calculations")
parser.add_argument("-i", action="store_true", default=False, help="Write an job script")
parser.add_argument(
        "program", type=str, choices=["dalton"], help="Name of the program in the python script."
    )
parser.add_argument("fnames", nargs="*", help="Path to existing input file(s).")
parser.add_argument("-ia", "--inactive", type=int, default=0, help="Number of inactive geminals. Default: 0.")
parser.add_argument("-n", "--nelec", type=int, default=False, help="Number of electrons.")
parser.add_argument("-t", "--temp", type=str, default=False, help="Template file.")
parser.add_argument("-b", "--basis", type=str, default=False, help="Basis set.")
parser.add_argument("-m", "--method", type=str, default=False, help="Method.")


def submit_serial_job(args):
    program = args.program.lower()
    if program != 'dalton':
        raise NotImplementedError(f"Input for program {program} not supported.")
    folders = sorted(args.fnames)

    for index, fp_job in enumerate(folders):
        print(f"Submit {index} {fp_job}")
        
        # get base directory
        base_database = os.getcwd()
        
        # get folder & filename
        folder, fn_job = os.path.split(fp_job)
        
        if folder != "":
            os.chdir(folder)
        
        # run job
        base_name = fn_job.split('.')[0]
        os.system(f"dalton_new {base_name}")
        os.system("sleep 3")

        os.chdir(base_database)


def make_input(args):
    program = args.program.lower()
    if program != 'dalton':
        raise NotImplementedError(f"Input for program {program} not supported.")
    if not args.nelec % 2 == 0:
        raise ValueError("Only even number of electrons supported.")
    basis = args.basis
    method = args.method
    if not method == 'gvbpp':
        raise NotImplementedError(f"Method {method} not supported.")
    basisname = basis.lower().replace("-", "").replace("*", "p").replace("+", "d")  

    npairs = args.nelec // 2
    NGEM = npairs - args.inactive
    params = {"inactive": args.inactive, "ngem": NGEM}

    with open(args.temp, 'r') as f:
        content = f.read()
    template = Template(content)    

    fp_mols = args.fnames
    for index, fp_mol in enumerate(fp_mols):
        print(f"Input {index} {fp_mol}")
        # get folder & filename
        folder, molfile = os.path.split(fp_mol)
        assert molfile.endswith('.mol')
        f_name = molfile.strip('.mol').split('_')[0]
        # HARDCODED
        charge = 0
        subdir = f'{f_name}_q00{charge}_m01_k00_sp_{method}_{basisname}'

        # get base directory
        base_database = os.getcwd()

        # Make job folder
        if folder == "":
            folder = '.'
        fp_job = f'{folder}/{subdir}'
        if not os.path.exists(fp_job):
            os.makedirs(fp_job)
        cp_mol = f'{fp_job}/{f_name}.mol'
        if not os.path.exists(cp_mol):
            os.system(f"cp {fp_mol} {cp_mol}")
        os.chdir(fp_job)

        string = template.substitute(params)
        # write input file
        with open(f'{f_name}.dal', 'w') as f:
            f.write(string)
        
        os.chdir(base_database)


if __name__ == "__main__":

    # Parse arguments
    args = parser.parse_args()

    # Exit if no commands are specified
    if not (args.r or args.i):
        print("No command specified. Exiting...", file=stderr)
        exit(1)

    # Run specified command(s)
    if args.r:
        submit_serial_job(args)
    if args.i:
        if not args.nelec:
            raise ValueError("The number of electrons must be specified.")
        if not args.basis:
            raise ValueError("The basis set must be specified.")
        if not args.method:
            raise ValueError("A method must be specified.")
        make_input(args)

    # Exit successfully
    exit(0)