"""Command-line interface for yunta."""
from typing import Callable
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
    outputs = fn(msa1, msa2, **kwargs)
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
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        max_recycles=args.recycles,
        param_dir=args.params,
        subbatch_size=args.subbatch_size,
        unified_memory=args.unified_memory,
        adaptive_msa=not args.fixed_msa_padding,
        recycle_tolerance=args.recycle_tolerance,
    )


@clicommand(message="Modelling sets of PPIs with the following parameters")
def _af2_many_vs_many(args: Namespace) -> None:
    from .screening import model_many_vs_many
    return _base_command(
        args,
        model_many_vs_many,
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        max_recycles=args.recycles,
        param_dir=args.params,
        subbatch_size=args.subbatch_size,
        unified_memory=args.unified_memory,
        adaptive_msa=not args.fixed_msa_padding,
        recycle_tolerance=args.recycle_tolerance,
    )


def _read_msa_args(args: Namespace) -> list:
    """Expand ``msa1`` / ``--msa2`` into a list of ``(MSA, MSA | None)`` pairs."""
    from .screening import _as_msa, _unordered_pairs
    msa1, msa2 = _msa_from_list_file(args)
    msa1 = msa1 if isinstance(msa1, list) else [msa1]
    cache = {}
    if msa2:
        msa2 = msa2 if isinstance(msa2, list) else [msa2]
        return [(_as_msa(a, cache), _as_msa(b, cache)) for a in msa1 for b in msa2]
    if len(msa1) == 1:  # a single MSA: homodimer
        return [(_as_msa(msa1[0], cache), None)]
    return [
        (_as_msa(a, cache), _as_msa(b, cache) if b is not None else None)
        for a, partners in _unordered_pairs(msa1) for b in partners
    ]


def _pairing_kwargs(args: Namespace) -> dict:
    return dict(
        interaction_map="builtin" if args.interspecies else None,
        enforce_ref_match=args.strict_match,
        pairing=args.pairing,
        seed=args.seed,
        max_gap_fraction=args.max_gap_fraction,
        strict_controls=not args.allow_invalid_controls,
    )


def _safe_name(name: str) -> str:
    """Make one *file name* safe (no path separators). Never apply to a whole path."""
    return name.replace(os.sep, "_").replace("/", "_")


def _sidecar_path(a3m_path: str) -> str:
    return os.path.splitext(a3m_path)[0] + ".json"


@clicommand(message="Pairing MSAs with the following parameters")
def _pair(args: Namespace) -> None:
    import json
    from .interactions.runner import _pair_msas
    from .structs.pairing import InvalidControlError, log_rejections
    pairs = _read_msa_args(args)
    kwargs = _pairing_kwargs(args)
    os.makedirs(args.output, exist_ok=True)
    rejected, n_pairs = {}, 0
    for msa1, msa2 in pairs:
        n_pairs += 1
        try:
            paired = _pair_msas(msa1, msa2, blocked=args.blocked, **kwargs)
        except InvalidControlError as e:
            name = f"{msa1.name}/{msa2.name if msa2 is not None else msa1.name}"
            print_err(f"[WARN] Rejected {name}: {e}")
            rejected[name] = str(e)
            continue
        if paired is None:
            continue
        filename = os.path.join(args.output, _safe_name(f"{paired.name}.{args.pairing}.a3m"))
        with open(filename, "w") as f:
            paired.write(f)
        sidecar = {
            "yunta_version": __version__,
            "ids": list(paired.ids),
            "chain_a_length": paired.chain_a_length,
            "chain_b_length": paired.chain_b_length,
            "msa1": getattr(msa1, "name", None),
            "msa2": getattr(msa2 if msa2 is not None else msa1, "name", None),
            "msa1_depth": len(msa1),
            "msa2_depth": len(msa2) if msa2 is not None else len(msa1),
            "max_gap_fraction": args.max_gap_fraction,
            "interspecies": bool(args.interspecies),
            "pairing": paired.pairing_info.as_dict() if paired.pairing_info is not None else None,
        }
        with open(_sidecar_path(filename), "w") as f:
            json.dump(sidecar, f, indent=2, default=str)
        print_err(f"[INFO] Wrote {filename} (+ {os.path.basename(_sidecar_path(filename))})")
    log_rejections(args.pairing, n_pairs, rejected)
    return None


def _read_paired_inputs(args: Namespace) -> tuple[list, list]:
    """Load paired A3Ms and their ``yunta pair`` sidecars, without re-pairing."""
    import json
    from .structs.msa import PairedMSA
    msas, depths = [], []
    inputs = args.msa1 if isinstance(args.msa1, list) else [args.msa1]
    for handle in inputs:
        path = getattr(handle, "name", handle)
        sidecar = _sidecar_path(path)
        if not os.path.exists(sidecar):
            raise FileNotFoundError(
                f"--paired needs the sidecar written by `yunta pair`: {sidecar} (it holds the chain boundary)."
            )
        with open(sidecar) as f:
            meta = json.load(f)
        paired = PairedMSA.read(handle, meta["chain_a_length"])
        if meta.get("pairing"):
            from .structs.pairing import PairingReport
            paired.pairing_info = PairingReport(**meta["pairing"])
        msas.append(paired)
        depths.append((meta.get("msa1_depth", len(paired)), meta.get("msa2_depth", len(paired))))
    return msas, depths


@clicommand(message="Predicting co-evolution with the following parameters")
def _coevo(args: Namespace) -> None:
    import numpy as np
    from .coevo import CoevoPredictor
    from .io import write_metrics

    if args.paired:
        paired, depths = _read_paired_inputs(args)
        predictor = CoevoPredictor.from_paired(paired, depths, cpu=args.cpu, chunksize=args.chunksize)
    else:
        predictor = CoevoPredictor(
            _read_msa_args(args),
            cpu=args.cpu,
            chunksize=args.chunksize,
            **_pairing_kwargs(args),
        )
    print_err(f"[INFO] {len(predictor)} pairs prepared ({len(predictor.skipped)} skipped, {len(predictor.rejected)} of them rejected controls).")
    os.makedirs(args.output, exist_ok=True)
    model_kwargs = {
        "af2": {"model_kwargs": {
            "max_recycles": args.recycles, "param_dir": args.params,
            "subbatch_size": args.subbatch_size, "unified_memory": args.unified_memory,
            "adaptive_msa": not args.fixed_msa_padding,
            "recycle_tolerance": args.recycle_tolerance,
        }},
        "of3": {"model_kwargs": {"checkpoint": args.of3_checkpoint}},
        "rf2t": {}, "dca": {},
    }
    run_kwargs = {
        "dca": {"apc": args.apc},
        "af2": {"structure": bool(args.structures)},
        "rf2t": {},
        "of3": {"structure": bool(args.structures)},
    }
    with predictor:
        for method in args.method:
            metrics = []
            try:
                for pred in predictor.predict(method, **model_kwargs[method], **run_kwargs[method]):
                    metrics.append(pred.metrics)
                    base = _safe_name(f"{method}.{pred.metrics.ID}")
                    np.savez_compressed(
                        os.path.join(args.output, f"{base}.npz"),
                        matrix=pred.matrix, interaction=pred.interaction,
                    )
                    if args.structures and getattr(pred.metrics, "structure", None):
                        os.makedirs(args.structures, exist_ok=True)
                        with open(os.path.join(args.structures, f"{base}.pdb"), "w") as f:
                            f.write(pred.metrics.structure)
                    if args.plot is not None:
                        _plot_results(pred.matrix, pred.interaction, pred.metrics,
                                      output_dir=os.path.join(args.plot, method))
                write_metrics(metrics, filename=os.path.join(args.output, f"{method}.tsv"))
            finally:
                predictor.close_model(method)  # one backend resident at a time, even on error
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
        help='MSAs are from different species, enables built-in host-pathogen interaction map. Default: Not inter-species.',
    )
    params = CLIOption('--params', '-w', 
                       type=str,
                       default=None,
                       help='Path to AlphaFold2 params file (.npz).')
    recycles = CLIOption('--recycles', '-x', 
                       type=int,
                       default=10,
                       help='Maximum number of recyles through the model.')

    pairing = CLIOption('--pairing', type=str, default='map',
                        choices=['map', 'rewired', 'random-species', 'random-rows', 'permute-partners'],
                        help='How to pair sequences. "map" = real pairing; the others are negative controls '
                             '(degree-preserving rewired species pairs, random species pairs, random rows, or '
                             'permuted B partners of the exact real rows, which keeps each chain\'s composition and gaps).')
    allow_invalid_controls = CLIOption('--allow-invalid-controls', action='store_true',
                                       help='Score negative controls that violate their invariants (depth, no known pairs, '
                                            'multiplicity, final depth after gap filtering) instead of rejecting them. '
                                            'They are flagged pairing_valid=False; exclude those from analyses. '
                                            'Default: invalid controls are rejected and listed.')
    paired_input = CLIOption('--paired', action='store_true',
                             help='Inputs are paired A3Ms written by `yunta pair` (with their .json sidecars): '
                                  'predict from exactly that pairing, without re-pairing.')
    seed = CLIOption('--seed', type=int, default=None,
                     help='Random seed for control pairings and AF2. Default: random (recorded in outputs).')
    max_gap_fraction = CLIOption('--max-gap-fraction', type=float, default=.9,
                                 help='Drop paired rows with a larger gap fraction. Default: 0.9.')
    blocked = CLIOption('--blocked', action='store_true',
                        help='Append unpaired sequences as blocks below the paired rows.')
    methods = CLIOption('--method', '-m', nargs='+', default=['dca'],
                        choices=['dca', 'rf2t', 'af2', 'of3'],
                        help='Methods to run, in order; each model is loaded once. Default: dca.')
    chunksize = CLIOption('--chunksize', type=int, default=None,
                          help='Opt in to chunked (blockwise) inference of at most this many alignment columns per '
                               'model call. Default: off, the whole complex is modelled in one call. Chunked scores '
                               'are approximations (score_mode=blockwise) and chunked complexes have no structure. '
                               'Try --subbatch-size / --unified-memory (AF2) before chunking.')
    output_dir = CLIOption('--output', '-o', type=str, required=True, help='Output directory.')
    subbatch_size = CLIOption('--subbatch-size', type=int, default=None,
                              help='AF2: attention/outer-product sub-batch size (stock 4). Smaller uses less memory, '
                                   'slower.')
    unified_memory = CLIOption('--unified-memory', action='store_true',
                               help='AF2 on GPU: let device memory spill to host RAM so larger complexes run '
                                    '(slowly) instead of failing.')
    recycle_tolerance = CLIOption('--recycle-tolerance', type=float, default=None,
                                  help='AF2: stop recycling early once the RMS change of the C-alpha distance matrix '
                                       'between two passes is below this many Angstrom (approximate; 0 runs every '
                                       'recycle and gives the stock result). Default: always run all recycles.')
    fixed_msa_padding = CLIOption('--fixed-msa-padding', action='store_true',
                                  help='AF2: pad the MSA to the stock 512 clusters / 5120 extra rows instead of the '
                                       'alignment\'s real depth (same results, more memory and time).')

    of3_checkpoint = CLIOption('--of3-checkpoint', type=str, default=None,
                               help='OpenFold3 checkpoint (.pt). Default: download the default weights on first use '
                                    'to $OPENFOLD_CACHE or ~/.openfold3.')
    structures = CLIOption('--structures', type=str, default=None,
                           help='Directory for predicted structures (PDB, pLDDT in the B-factor column) from AF2 and '
                                'OpenFold3, named <method>.<pair>.pdb. Only for complexes modelled in one call '
                                '(i.e. not chunked).')

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
        options=[inputs] + base_opts + [params, recycles, subbatch_size, unified_memory, recycle_tolerance, fixed_msa_padding],
    )
    af2_many = CLICommand(
        'af2-many', 
        description='Model all interactions between two sets of proteins, or all pairs in one set of proteins.',
        main=_af2_many_vs_many,
        options=[inputs_list] + base_opts + [params, recycles, subbatch_size, unified_memory, recycle_tolerance, fixed_msa_padding],
    )

    pair_cmd = CLICommand(
        'pair',
        description='Write paired MSAs (real pairing or a negative control) as A3M files.',
        main=_pair,
        options=[inputs_list, inputs_list2, list_file, interspecies, strict_match, pairing, seed,
                 allow_invalid_controls, max_gap_fraction, blocked, output_dir],
    )
    coevo_cmd = CLICommand(
        'coevo',
        description='Predict inter-protein interactions with dca, rf2t and/or af2, for all pairs of the inputs.',
        main=_coevo,
        options=[inputs_list, inputs_list2, list_file, interspecies, strict_match, pairing, seed,
                 allow_invalid_controls, paired_input, max_gap_fraction, methods, chunksize, apc, params, recycles, subbatch_size, unified_memory, recycle_tolerance, fixed_msa_padding, cpu, plot, structures, of3_checkpoint, output_dir],
    )

    app = CLIApp(
        appname,
        version=__version__,
        description="Screening protein-protein interactions using DCA, RosettaFold-2track, and AlphaFold2.",
        commands=[pair_cmd, coevo_cmd, dca_single, dca_many, rf2t_single, af2_single, af2_many],
    )

    app.run()
    return None


if __name__ == "__main__":
    main()