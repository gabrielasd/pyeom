import os, sys 
from glob import glob

from string import Template

from argparse import ArgumentParser

from sys import stderr


# Initialize command line argument parser
parser = ArgumentParser()


# Specify positional arguments and options
parser.add_argument("-r", action="store_true", default=False, help="submit calculations")
parser.add_argument("-i", action="store_true", default=False, help="Make slurm script")
parser.add_argument(
        "program", type=str, choices=["erpa", "gqcp", "pyscf"], help="Name of the program in the python script."
    )
parser.add_argument(
    "job", type=str, choices=["ecor", "exc", "sp"], 
    help="Job type (ecor: ground state energy correction, exc: excitation energy, sp: single point)."
)
parser.add_argument("fnames", nargs="*", help="Path to existing input file(s).")
parser.add_argument("-time", help="Allocated time for the job in the format HH:MM:SS.")
parser.add_argument(
    "-a", "--account", type=str, choices=["def-ayers", "rrg-ayers-ab"], default="def-ayers",
    help="Account charged for resources used."
    )
parser.add_argument("-m", "--memory", type=int, default=4, help="Allocated memory for the job in GB. Default: 4 GB.")
parser.add_argument("-t", "--temp", type=str, default=False, help="Template file.")
parser.add_argument("-p", "--nprocs", type=int, default=1, help="Number of processors.")


def submit_slurm(args):
    program = args.program.lower()
    folders = sorted(args.fnames)

    # get base directory
    base_database = os.getcwd()
    for index, fp_job in enumerate(folders):
        print(f"Submit {index} {fp_job}")
        if not fp_job.endswith(".sh"):
            raise ValueError(f"Expected bash script to have .sh extension, got {fp_job}")
        
        # get folder & filename
        folder, fn_job = os.path.split(fp_job)           
        
        if folder != "":
            os.chdir(folder)
        
        if program == 'erpa':
            name = fn_job.split('.')[0]
            if args.job == 'ecor':
                assert name.endswith('_ac')
            elif args.job == 'exc':
                assert name.endswith('_exc')
            else:
                raise ValueError(f"Job {args.job} not supported with program {program}.")
        
        # run job
        os.system(f"sbatch {fn_job}")
        os.system("sleep 3")

        os.chdir(base_database)


def make_slurm(args):
    program = args.program.lower()    
    params = {"time": args.time, "memory": args.memory, "n_proc": args.nprocs, "account": args.account }
    folders = sorted(args.fnames)    

    if program not in args.temp:
        raise ValueError(f"Template file {args.temp} does not match program {program}.")
    with open(args.temp, 'r') as f:
        content = f.read()
    template = Template(content)    

    # get base directory
    base_database = os.getcwd()

    for index, fp_job in enumerate(folders):
        print(f"Input {index} {fp_job}")
        # get folder & filename
        folder, subdir = os.path.split(fp_job)

        params['name'] = f'{subdir}'
        string = template.substitute(params)

        os.chdir(fp_job)
        # write input file
        if args.program == "erpa":
            if args.job == 'ecor':
                ofilename = f'{subdir}_ac.sh'
            elif args.job == 'exc':
                ofilename = f'{subdir}_exc.sh'
            else:
                raise ValueError(f"Job {args.job} not supported.")
        else:
            ofilename = f'{subdir}.sh'
        with open(ofilename, 'w') as f:
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
        submit_slurm(args)
    if args.i:
        make_slurm(args)

    # Exit successfully
    exit(0)