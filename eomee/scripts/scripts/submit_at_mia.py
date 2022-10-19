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
        "program", type=str, choices=["pyci", "erpa", "gqcp"], help="Name of the program in the python script."
    )
parser.add_argument("fnames", nargs="*", help="Path to existing input file(s).")
parser.add_argument("-c", "--charge", type=int, default=0, help="Charge of the molecule. Default: 0.")
parser.add_argument("-m", "--mult", type=int, default=1, help="Multiplicity of the molecule. Default: 1.")
parser.add_argument("-t", "--temp", type=str, default=False, help="Template file.")
parser.add_argument("-n", "--nelec", type=int, default=0, help="Number of electrons.")
parser.add_argument("-l", "--lot", type=str, default="doci", help="CI wavefunction type.")
parser.add_argument("-p", "--nprocs", type=int, default=1, help="Number of processors.")


def submit_serial_job(args):
    program = args.program.lower()
    folders = sorted(args.fnames)

    for index, fp_job in enumerate(folders):
        print(f"Submit {index} {fp_job}")
        if not fp_job.endswith(".py"):
            raise ValueError(f"Expected Python script to have .py extension, got {fp_job}")
        
        # get base directory
        base_database = os.getcwd()
        
        # get folder & filename
        folder, fn_job = os.path.split(fp_job)

        if program == 'erpa':
            name = fn_job.split('.')[0]
            assert name.endswith('_ac')
        
        if folder != "":
            os.chdir(folder)
        
        # run job
        os.system(f"python3 {fn_job}")
        os.system("sleep 3")

        os.chdir(base_database)


def make_input(args):
    program = args.program.lower()
    if program != 'erpa':
        raise NotImplementedError(f"Input for program {program} not supported.")
    
    params = {"charge": args.charge, "spinmult": args.mult, "nelec": args.nelec, "lot": args.lot, "nprocs": args.nprocs}
    folders = sorted(args.fnames)    

    with open(args.temp, 'r') as f:
        content = f.read()
    template = Template(content)    

    for index, fp_job in enumerate(folders):
        print(f"Input {index} {fp_job}")
        # get folder & filename
        folder, subdir = os.path.split(fp_job)

        params['title'] = f'{fp_job}'
        params['output'] = f'{subdir}'
        string = template.substitute(params)

        # get base directory
        base_database = os.getcwd()
        os.chdir(fp_job)

        # write input file
        with open(f'{subdir}_ac.py', 'w') as f:
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
        make_input(args)

    # Exit successfully
    exit(0)