import os, os.path as osp, logging, uuid, fnmatch, pprint
from contextlib import contextmanager

import numpy as np

VERSION = '0.1'


def uid():
    return str(uuid.uuid4())


def setup_logger(name='hgcal-npz'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[92m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
logger = setup_logger()


def is_remote(path):
    """
    Checks whether a file is remote
    """
    return '://' in path


def expand_wildcards(pats):
    """
    Expands wildcards, also for remote paths with a wildcard
    """
    expanded = []
    for pat in pats:
        if '*' in pat:
            if is_remote(pat):
                import seutils
                expanded.extend(seutils.ls_wildcard(pat))
            else:
                import glob
                expanded.extend(glob.glob(pat))
        else:
            expanded.append(pat)
    return expanded


@contextmanager
def local_copy(remote):
    """
    Creates a temporary local copy of a remote file
    """
    must_delete = False
    try:
        if is_remote(remote):
            # File is remote, make local copy
            import seutils
            must_delete = True
            local = uid() + osp.splitext(remote)[1]
            logger.info('Copying %s -> %s', remote, local)
            seutils.cp(remote, local)
            yield local
        else:
            # File is already local, nothing to do
            yield remote
    finally:
        if must_delete:
            try:
                os.remove(local)
            except Exception:
                pass


class DataFrame:
    """
    Poor man's data frame: just stores a rectangular array,
    and a list of strings about what the columns are
    """
    def __init__(self, keys=[]):
        self.keys = keys
        self.array = None

    def __len__(self):
        return len(self.array)

    def __eq__(self, other):
        return self.keys == other.keys and np.array_equal(self.array, other.array)


class Event:
    """
    Collection of several dataframes that describe the event
    """

    @classmethod
    def load(cls, infile):
        inst = cls()
        with local_copy(infile) as local:
            d = np.load(local, allow_pickle=True)
        inst.metadata = d['metadata'].item()
        inst.metadata['npz_file'] = infile
        inst.rechits.keys = list(d['rechits_keys'])
        inst.rechits.array = d['rechits_array']
        inst.simtracks.keys = list(d['simtracks_keys'])
        inst.simtracks.array = d['simtracks_array']
        inst.simclusters.keys = list(d['simclusters_keys'])
        inst.simclusters.array = d['simclusters_array']
        return inst

    def __init__(self):
        self.metadata = {'version': VERSION}
        self.rechits = DataFrame()
        self.simtracks = DataFrame()
        self.simclusters = DataFrame()

    def save(self, outfile):
        do_stageout = False
        if is_remote(outfile):
            import seutils
            remote_outfile = outfile
            outfile = uid() + '.npz'
            do_stageout = True
        logger.info('Dumping to %s', outfile)
        # Automatically create parent directory if not existent
        outdir = osp.dirname(osp.abspath(outfile))
        if not osp.isdir(outdir):
            os.makedirs(outdir)

        np.savez(
            outfile,
            metadata = self.metadata,
            rechits_keys = self.rechits.keys,
            rechits_array = self.rechits.array,
            simtracks_keys = self.simtracks.keys,
            simtracks_array = self.simtracks.array,
            simclusters_keys = self.simclusters.keys,
            simclusters_array = self.simclusters.array,
            )

        if do_stageout:
            logger.info('Staging out %s -> %s', outfile, remote_outfile)
            seutils.cp(outfile, remote_outfile)
            os.remove(outfile)


def events_factory(rootfile):
    """
    Iterates Event instances from a rootfile.
    """
    branches = [
        'RecHitHGC_x',
        'RecHitHGC_y',
        'RecHitHGC_z',
        'RecHitHGC_energy',
        'RecHitHGC_time',
        'RecHitHGC_MergedSimClusterBestMatchIdx',
        'RecHitHGC_MergedSimClusterBestMatchQual',

        'MergedSimCluster_boundaryEnergy',
        'MergedSimCluster_boundaryP4',
        'MergedSimCluster_recEnergy',
        'MergedSimCluster_pdgId',
        'MergedSimCluster_trackIdAtBoundary',

        'SimTrack_pdgId',
        'SimTrack_trackId',
        'SimTrack_boundaryMomentum_eta',
        'SimTrack_boundaryMomentum_phi',
        'SimTrack_boundaryMomentum_pt',
        'SimTrack_boundaryPos_x',
        'SimTrack_boundaryPos_y',
        'SimTrack_boundaryPos_z',
        'SimTrack_eta',
        'SimTrack_phi',
        'SimTrack_pt',
        'SimTrack_trackIdAtBoundary',
        'SimTrack_crossedBoundary',
        ]
    
    import uproot
    with local_copy(rootfile) as local:
        with uproot.open(local + ':Events') as t:
            hlarray = t.arrays(branches)

    rechit_keys = fnmatch.filter(branches, 'RecHitHGC_*')
    simtrack_keys = fnmatch.filter(branches, 'SimTrack_*')
    simcluster_keys = fnmatch.filter(branches, 'MergedSimCluster_*')

    for i in range(len(hlarray)):
        event = Event()
        event.metadata['rootfile'] = rootfile
        event.metadata['i_event'] = i
        event.rechits.keys = rechit_keys
        event.simtracks.keys = simtrack_keys
        event.simclusters.keys = simcluster_keys

        # Build the rectangular array from the keys
        for df in [event.rechits, event.simtracks, event.simclusters]:
            df.array = np.stack([hlarray[key][i].to_numpy() for key in df.keys]).T

        yield event


def cli_produce_worker(tup):
    """
    Multiprocessing worker for cli_produce
    """
    args, rootfile = tup
    n_done = 0
    for event in events_factory(rootfile):
        if args.n and n_done==args.n: return
        dst = osp.join(
            args.outdir,
            osp.basename(rootfile).replace('.root', f'_{event.metadata["i_event"]:03d}.npz')
            )
        event.save(dst)
        n_done += 1


def cli_produce():
    """
    Command line interface
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rootfiles', type=str, nargs='+', help='.root files to produce .npz files from')
    parser.add_argument('-d', '--outdir', type=str, help='Destination directory', required=True)
    parser.add_argument('-n', type=int, help='Max nr. of events to process (default=all)')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of simultaneous processes (does one rootfile per process)')
    args = parser.parse_args()

    if not is_remote(args.outdir) and not osp.isdir(args.outdir): os.makedirs(args.outdir)
    if args.n and args.nthreads>1: logger.warning('-n works PER THREAD!')

    mp_args = [(args, rootfile) for rootfile in expand_wildcards(args.rootfiles)]
    import multiprocessing as mp
    p = mp.Pool(args.nthreads)
    p.map(cli_produce_worker, mp_args)


def cli_ls():
    """
    Command line interface for quickly printing some contents from a .npz
    file to the terminal
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfiles', type=str, nargs='+', help='.npz files to print')
    args = parser.parse_args()

    def format_and_indent(obj):
        f = pprint.pformat(obj, indent=6)
        return f[0] + '\n ' + f[1:]

    for npzfile in args.npzfiles:
        event = Event.load(npzfile)
        print(
            f'<Event'
            f'\n  metadata={format_and_indent(event.metadata)}'
            f'\n  rechits:'
            f'\n    shape={event.rechits.array.shape}'
            f'\n    keys={format_and_indent(event.rechits.keys)}'
            f'\n  simtracks:'
            f'\n    shape={event.simtracks.array.shape}'
            f'\n    keys={format_and_indent(event.simtracks.keys)}'
            f'\n  simclusters:'
            f'\n    shape={event.simclusters.array.shape}'
            f'\n    keys={format_and_indent(event.simclusters.keys)}'
            )
 