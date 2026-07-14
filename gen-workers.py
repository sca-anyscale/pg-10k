#!/usr/bin/env python3
import argparse
from jinja2 import Environment, FileSystemLoader


def main():
    ''' main '''
    parser = argparse.ArgumentParser(description='generate rack labels')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--nodes-per-rack', type=int, default=18)
    parser.add_argument('-r', '--num-racks', type=int, default=111)

    args = parser.parse_args()

    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    template = env.get_template('job.yaml.in')

    props = {
        'racks': args.num_racks,
        'nodes_per_rack': args.nodes_per_rack,
        'total_nodes': args.num_racks * args.nodes_per_rack,
    }
    output = template.render(props)

    with open('job.yaml', 'w', encoding='utf-8') as ofile:
        ofile.write(output)
        ofile.write('\n')

    template = env.get_template('RUNIT.in')
    output = template.render(props)

    with open('RUNIT', 'w', encoding='utf-8') as ofile:
        ofile.write(output)
        ofile.write('\n')

if __name__ == '__main__':
    main()
