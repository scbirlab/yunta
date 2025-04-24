"""Command-line interface for yunta."""

from argparse import FileType, Namespace
import os
import sys

from carabiner import cast, print_err
from carabiner.cast import flatten
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp

from . import __version__
from .io import write_metrics
from .plots import plot_matrix
from .screening import (
    dca_one_vs_many, 
    dca_many_vs_many, 
    model_one_vs_many, 
    model_many_vs_many, 
    rf2track_one_vs_many,
)

def _load_msa_list(*args):
    args = [a[0] if isinstance(a, list) else a for a in args]
    return [
        flatten([line.strip() for line in msa]) 
        for msa in args
    ]

def _plot_results(results, result_interaction, metric, 
                  output_dir: str = '.', *args, **kwargs) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename_prefix = os.path.join(output_dir, metric.ID)
    if hasattr(metric, 'apc'):
        apc = metric.apc
        filename_prefix += f".{apc=}"    
    plot_matrix(results, 
                filename_prefix=filename_prefix, 
                hline=metric.chain_a_len, 
                vline=metric.chain_a_len)
    plot_matrix(result_interaction, 
                filename_prefix=f"{filename_prefix}.interaction",
                ylabel=metric.uniprot_id_1, 
                xlabel=metric.uniprot_id_2)
    return None


def _msa_from_list_file(args: Namespace) -> tuple:
    if args.list_file:
        msa1, msa2 = _load_msa_list(args.msa1, args.msa2)
    else:
        msa1, msa2 = args.msa1, args.msa2
    return msa1, msa2


@clicommand(message="Making RosettaFold-2track prediction with the following parameters")
def _rf2t_single(args: Namespace) -> None:

    msa1, msa2 = _msa_from_list_file(args)

    print_err(f"Running RF-2t using {msa1} as reference.")
    outputs = rf2track_one_vs_many(
        msa_file1=msa1,
        msa_file2=msa2,
        cpu=args.cpu,
        interaction_map="builtin" if args.interspecies else None,
    )
    metrics = [_output[-1] for _output in outputs]
    write_metrics(metrics, 
                  filename=args.output)
    if args.plot is not None:
        for _output in outputs:
            _plot_results(*_output, output_dir=args.plot)

    return None


@clicommand(message="Calculating DCA for a pair of MSAs with the following parameters")
def _dca_single(args: Namespace) -> None:

    msa1, msa2 = _msa_from_list_file(args)

    outputs = dca_one_vs_many(
        msa_file1=msa1,
        msa_file2=msa2,
        apc=args.apc,
        interaction_map="builtin" if args.interspecies else None,
    )
    metrics = [_output[-1] for _output in outputs]
    write_metrics(metrics, 
                  filename=args.output)
    if args.plot is not None:
        for _output in outputs:
            _plot_results(*_output, output_dir=args.plot)

    return None


@clicommand(message="Calculating DCA between pairs of MSAs with the following parameters")
def _dca_many_vs_many(args: Namespace) -> None:

    msa1, msa2 = _msa_from_list_file(args)

    outputs = dca_many_vs_many(
        msa_files1=msa1,
        msa_files2=msa2,
        apc=args.apc,
        interaction_map="builtin" if args.interspecies else None,
    )

    metrics = [_output[-1] for _output in outputs]
    write_metrics(metrics, 
                  filename=args.output)
    if args.plot is not None:
        for _output in outputs:
            _plot_results(*_output, output_dir=args.plot)

    return None


@clicommand(message="Modelling one PPI with the following parameters")
def _af2_single(args: Namespace) -> None:

    msa1, msa2 = _msa_from_list_file(args)

    metric = model_one_vs_many(
        msa_file1=msa1,
        msa_file2=msa2,
        max_recycles=args.recycles,
        output_dir=args.output,
        param_dir=args.params,
        interaction_map="builtin" if args.interspecies else None,
    )

    output_filename = os.path.join(args.output, f"_all_metrics.tsv")
    print_err(f"Saving metrics as {output_filename}")
    write_metrics(metric, 
                  filename=output_filename)

    return None


@clicommand(message="Modelling sets of PPIs with the following parameters")
def _af2_many_vs_many(args: Namespace) -> None:

    msa1, msa2 = _msa_from_list_file(args)

    metrics = model_many_vs_many(
        msa_files1=msa1,
        msa_files2=msa2,
        output_dir=args.output,
        max_recycles=args.recycles,
        param_dir=args.params,
        interaction_map="builtin" if args.interspecies else None,
    )

    output_filename = os.path.join(args.output, "_all_metrics.tsv")
    print_err(f"Saving metrics as {output_filename}")
    write_metrics(metrics, 
                  filename=output_filename)

    return None


def main() -> None:
    inputs = CLIOption('msa1', 
                       default=sys.stdin,
                       type=FileType('r'), 
                       nargs='?',
                       help='MSA file. Default: STDIN.')
    input2 = CLIOption('--msa2', '-2', 
                       type=FileType('r'), 
                       default=None,
                       help='Second MSA file.')
    inputs_list = CLIOption('msa1', 
                            default=sys.stdin,
                            type=FileType('r'), 
                            nargs='*',
                            help='MSA file(s).')
    inputs_list2 = CLIOption('--msa2', '-2', 
                        type=FileType('r'), 
                        default=None,
                        nargs='*',
                        help='Second MSA file(s). Default: if not provided, all pairwise from msa1.')
    list_file = CLIOption('--list-file', '-l', 
                          action='store_true',
                          help='Treat inputs as plain-text list of MSA files, rather than MSA filenames. '
                               'Default: treat as MSA filenames.')
    output = CLIOption('--output', '-o', 
                       type=str,
                       required=True,
                       help='Output directory.')
    plot = CLIOption('--plot', '-p', 
                     type=str,
                     default=None,
                     help='Directory for saving plots. Default: don\'t plot.')
    cpu = CLIOption('--cpu', '-c', 
                    action='store_true',
                    help='Whether to use CPU only. Default: use GPU if available.')
    output_file = CLIOption('--output', '-o', 
                            default=sys.stdout,
                            type=FileType('w'), 
                            nargs='?',
                            help='Output filename. Default: STDOUT.')
    apc = CLIOption('--apc', '-a', 
                    action='store_true',
                    help='Whether to use APC correction in DCA. Default: don\'t apply correction.')
    interspecies = CLIOption(
        '--interspecies', '-i', 
        action='store_true',
        help='Whether the MSAs are from the same species. Default: Not inter-species.',
    )
    params = CLIOption('--params', '-w', 
                       type=str,
                       default=None,
                       help='Path to AlphaFold2 params file (.npz).')
    recycles = CLIOption('--recycles', '-x', 
                       type=int,
                       default=10,
                       help='Maximum number of recyles through the model.')

    rf2t_single = CLICommand('rf2t-single', 
                            description='Calculate RF-2track contacts for between one protein and a series of others.',
                            main=_rf2t_single,
                            options=[inputs, inputs_list2, list_file, interspecies, output_file, plot, cpu])
    dca_single = CLICommand('dca-single', 
                            description='Calculate DCA for one protein-protein interaction.',
                            main=_dca_single,
                            options=[inputs, inputs_list2, list_file, interspecies, output_file, plot, apc])
    dca_many = CLICommand('dca-many', 
                          description='Calculate DCA between two sets of proteins, or all pairs in one set of proteins.',
                          main=_dca_many_vs_many,
                          options=[inputs_list, inputs_list2, list_file, interspecies, apc, output_file, plot])
    af2_single = CLICommand('af2-single', 
                            description='Model one protein-protein interaction.',
                            main=_af2_single,
                            options=[inputs, inputs_list2, list_file, output, interspecies, params, recycles, plot])
    af2_many = CLICommand('af2-many', 
                          description='Model all interactions between two sets of proteins, or all pairs in one set of proteins.',
                          main=_af2_many_vs_many,
                          options=[inputs_list, inputs_list2, list_file, interspecies, output, params, recycles, plot])

    app = CLIApp(
        "yunta",
        version=__version__,
        description="Screening protein-protein interactions using DCA, RosettaFold-2track, and AlphaFold2.",
        commands=[dca_single, dca_many, rf2t_single, af2_single, af2_many],
    )

    app.run()
    return None


if __name__ == "__main__":
    main()