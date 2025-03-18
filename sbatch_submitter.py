from datetime import datetime
import os
import re
import sys

def create_sbatch_file(command, sbatch_filename):
    sbatch_content = f'''#!/bin/sh
#SBATCH --partition=insy
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/11.8 cudnn/11-8.6.0.163
{command}'''

    with open(sbatch_filename, 'w') as sbatch_file:
        sbatch_file.write(sbatch_content)

def submit_sbatch_file(sbatch_filename):
    os.system(f'sbatch {sbatch_filename}')


def format_filename(script_name):
    # Remove path if any
    script_name = os.path.basename(script_name)
    # Remove '.py' extension
    script_name = re.sub(r'\.py$', '', script_name)
    # Replace non-alphanumeric characters with underscores
    script_name = re.sub(r'[^a-zA-Z0-9]', '_', script_name)
    return script_name + '.sbatch'

if __name__ == "__main__":
    command = sys.argv[1]
    script_name = re.findall(r'python(?: -m)?\s+"?(.+?)(?:"|\s)', command)[0]
    sbatch_filename = format_filename(script_name)
    formatted_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    sbatch_filename = f"{formatted_time}_{sbatch_filename}"
    print(f'Creating .sbatch for command {command} at {sbatch_filename}')
    create_sbatch_file(command, sbatch_filename)
    submit_sbatch_file(sbatch_filename)