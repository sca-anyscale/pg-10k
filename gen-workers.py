#!/usr/bin/env python3
import argparse
from jinja2 import Environment, FileSystemLoader


def main():
    ''' main '''
    parser = argparse.ArgumentParser(description='generate rack labels')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--num-racks', type=int, default=111)

    args = parser.parse_args()

    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    template = env.get_template('job.yaml.in')

    props = {
        'racks': args.num_racks,
    }
    output = template.render(props)

    with open('job.yaml', 'w', encoding='utf-8') as ofile:
        ofile.write(output)
        ofile.write('\n')


if __name__ == '__main__':
    main()
