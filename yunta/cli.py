"""Command-line interface for yunta."""

from argparse import FileType, Namespace
import os
import sys

from carabiner import cast, print_err
from carabiner.cast import flatten
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp

from . import appname, __version__
from .io import write_metrics


def _load_msa_list(*args):
    args = [a[0] if isinstance(a, list) else a for a in args]
    return [
        flatten([line.strip() for line in msa]) 
        for msa in args
    ]

def _plot_results(results, result_interaction, metric, 
                  output_dir: str = '.', *args, **kwargs) -> None:
    from .plots import plot_matrix
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


def _base_command(args: Namespace, fn: Callable, **kwargs):
    msa1, msa2 = _msa_from_list_file(args)
    outputs = fn(**kwargs)
    metrics = [_output[-1] for _output in outputs]
    write_metrics(
        metrics, 
        filename=args.output,
    )
    if args.plot is not None:
        for _output in outputs:
            _plot_results(
                *_output, 
                output_dir=args.plot,
            )
    return None

@clicommand(message="Making RosettaFold-2track prediction with the following parameters")
def _rf2t_single(args: Namespace) -> None:
    from .screening import rf2track_one_vs_many
    return _base_command(
        args,
        rf2track_one_vs_many,
        msa_file1=msa1,
        msa_file2=msa2,
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        cpu=args.cpu,
    )


@clicommand(message="Calculating DCA for a pair of MSAs with the following parameters")
def _dca_single(args: Namespace) -> None:
    from .screening import dca_one_vs_many
    return _base_command(
        args,
        dca_one_vs_many,
        msa_file1=msa1,
        msa_file2=msa2,
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        apc=args.apc,
    )


@clicommand(message="Calculating DCA between pairs of MSAs with the following parameters")
def _dca_many_vs_many(args: Namespace) -> None:
    from .screening import dca_many_vs_many
    return _base_command(
        args,
        dca_many_vs_many,
        msa_files1=msa1,
        msa_files2=msa2,
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        apc=args.apc,
    )


@clicommand(message="Modelling one PPI with the following parameters")
def _af2_single(args: Namespace) -> None:
    from .screening import model_one_vs_many
    return _base_command(
        args,
        model_one_vs_many,
        msa_file1=msa1,
        msa_file2=msa2,
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        max_recycles=args.recycles,
        param_dir=args.params,
    )


@clicommand(message="Modelling sets of PPIs with the following parameters")
def _af2_many_vs_many(args: Namespace) -> None:
    from .screening import model_many_vs_many
    return _base_command(
        args,
        model_many_vs_many,
        msa_file1=msa1,
        msa_file2=msa2,
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        max_recycles=args.recycles,
        param_dir=args.params,
    )


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
    strict_match = CLIOption(
        '--strict-match', '-S', 
        action='store_true',
        help='For interspecies, whether query MSA lines should be known interacting species.',
    )
    
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
        help='MSAs are from differnt species, enables built-in host-pathogen interaction map. Default: Not inter-species.',
    )
    params = CLIOption('--params', '-w', 
                       type=str,
                       default=None,
                       help='Path to AlphaFold2 params file (.npz).')
    recycles = CLIOption('--recycles', '-x', 
                       type=int,
                       default=10,
                       help='Maximum number of recyles through the model.')

    base_opts = [inputs_list2, list_file, interspecies, strict_match, output_file, plot]

    rf2t_single = CLICommand(
        'rf2t-single', 
        description='Calculate RF-2track contacts between one protein and a series of others.',
        main=_rf2t_single,
        options=[inputs] + base_opts + [cpu],
    )
    dca_single = CLICommand(
        'dca-single', 
        description='Calculate DCA for one protein-protein interaction.',
        main=_dca_single,
        options=[inputs] + base_opts + [apc],
    )
    dca_many = CLICommand(
        'dca-many', 
        description='Calculate DCA between two sets of proteins, or all pairs in one set of proteins.',
        main=_dca_many_vs_many,
        options=[inputs_list] + base_opts + [apc],
    )
    af2_single = CLICommand(
        'af2-single', 
        description='Model one protein-protein interaction.',
        main=_af2_single,
        options=[inputs] + base_opts + [params, recycles],
    )
    af2_many = CLICommand(
        'af2-many', 
        description='Model all interactions between two sets of proteins, or all pairs in one set of proteins.',
        main=_af2_many_vs_many,
        options=[inputs_list] + base_opts + [params, recycles],
    )

    app = CLIApp(
        appname,
        version=__version__,
        description="Screening protein-protein interactions using DCA, RosettaFold-2track, and AlphaFold2.",
        commands=[dca_single, dca_many, rf2t_single, af2_single, af2_many],
    )

    app.run()
    return None


if __name__ == "__main__":
    main()