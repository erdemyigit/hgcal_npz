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

    def copy(self):
        copy = DataFrame()
        copy.keys = self.keys.copy()
        copy.array = self.array[:]
        return copy

    def to_numpy(self, columns=None):
        """
        Returns a numpy array of the requested columns.
        If `columns` is None, all columns are returned
        """
        if columns is None: return self.array
        selected_col_idxs = np.array([self.keys.index(c) for c in columns])
        return self.array[:,selected_col_idxs]

    def get(self, key):
        """
        Gets a single column
        """
        return self.array[:, self.keys.index(key)].flatten()

    def add_column(self, key, vals):
        """
        Adds a column
        """
        assert key not in self.keys
        self.keys.append(key)
        self.array = np.column_stack((self.array, np.array(vals)))


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

    def copy(self):
        copy = Event()
        copy.metadata = self.metadata.copy()
        copy.rechits = self.rechits.copy()
        copy.simtracks = self.simtracks.copy()
        copy.simclusters = self.simclusters.copy()
        return copy

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

        'MergedSimCluster_impactPoint_eta',
        'MergedSimCluster_impactPoint_phi',
        'MergedSimCluster_impactPoint_x',
        'MergedSimCluster_impactPoint_y',
        'MergedSimCluster_impactPoint_z',
        'MergedSimCluster_boundaryEnergy',
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

        # Filter out hits with z=0 - not sure why these appear
        z_zero = event.rechits.get('RecHitHGC_z') == 0.
        n_z_zero_hits = z_zero.sum()
        if n_z_zero_hits:
            logger.warning(f'Filtering out {n_z_zero_hits} hits with z=0')
            event.rechits.array = event.rechits.array[~z_zero]

        pos, neg = split(event, fnmatch.filter(branches, '*_z'))
        flip(neg, fnmatch.filter(branches, '*_z')+fnmatch.filter(branches, '*_eta'))
        augment(pos)
        augment(neg)

        yield pos
        yield neg


def split(event, z_split_branches):
    """
    Takes a single event with hits/tracks/clusters in both endcaps,
    and splits it up into two 'events', one per endcap
    """
    pos = event.copy()
    pos.metadata['endcap'] = 'pos'
    neg = event.copy()
    neg.metadata['endcap'] = 'neg'

    for e in [pos, neg]:
        for df in [e.rechits, e.simclusters, e.simtracks]:
            for z_branch in z_split_branches:
                if z_branch in df.keys:
                    # Found a splittable branch; use it, and don't split
                    # the dataframe any further
                    mask = df.get(z_branch) >= 0. if e.metadata['endcap']=='pos' else df.get(z_branch) < 0.
                    df.array = df.array[mask]
                    break
    
    logger.info(
        f'Split event {event.metadata}:'
        f' {len(event.rechits)} hits to {len(pos.rechits)} in pos, {len(neg.rechits)} in neg;'
        f' {len(event.simtracks)} simtracks to {len(pos.simtracks)} in pos, {len(neg.simtracks)} in neg;'
        f' {len(event.simclusters)} simclusters to {len(pos.simclusters)} in pos, {len(neg.simclusters)} in neg;'
        )
    return pos, neg


def flip(event, z_flip_branches):
    """
    Multiplies all specified branches by -1.
    """
    for df in [event.rechits, event.simclusters, event.simtracks]:
        for z_branch in z_flip_branches:
            if z_branch in df.keys: df.array[:,df.keys.index(z_branch)] *= -1.


def augment(event: Event):
    """
    Adds some redundant but helpful-for-training information to the event.
    """
    # Some augmentation: add spherical coordinates on top of cartesion ones for rechits
    x = event.rechits.get('RecHitHGC_x')
    y = event.rechits.get('RecHitHGC_y')
    z = event.rechits.get('RecHitHGC_z')

    T = np.sqrt(x**2+y**2) # distance in Transverse plane
    theta = np.arctan2(T, z)
    eta = -np.log(np.tan(theta/2))

    bad_rows = np.concatenate((
        np.nonzero(~np.isfinite(theta))[0],
        np.nonzero(~np.isfinite(eta))[0]
        ))
    if len(bad_rows):
        logger.error(
            f'Found non-finite numbers at indices {bad_rows}:\n'
            f'{event.rechits.array[bad_rows]}'
            )
        raise Exception

    R = np.sqrt(x**2+y**2+z**2)
    phi = np.arctan2(x, y) # Assuming phi=0 points vertically upward
    # Map to -pi ... pi
    phi %= 2.*np.pi
    phi[phi > np.pi] -= 2.*np.pi

    event.rechits.add_column('RecHitHGC_theta', theta)
    event.rechits.add_column('RecHitHGC_eta', eta)
    event.rechits.add_column('RecHitHGC_phi', phi)
    event.rechits.add_column('RecHitHGC_R', R)

    # Add a sensible cluster number
    y = event.rechits.get('RecHitHGC_MergedSimClusterBestMatchIdx')
    y_incremental = incremental_cluster_index(y, noise_index=-1)
    event.rechits.add_column('RecHitHGC_incrClusterIdx', y_incremental)

    # Sort rechits by this sensible cluster number
    order = y_incremental.argsort()
    event.rechits.array = event.rechits.array[order]


def incremental_cluster_index(input: np.array, noise_index=None):
    """
    Build a map that translates arbitrary indices to ordered starting from zero

    By default the first unique index will be 0 in the output, the next 1, etc.
    E.g. [13 -1 -1 13 -1 13 13 42 -1 -1] -> [0 1 1 0 1 0 0 2 1 1]

    If noise_index is not None, the output will be 0 where input==noise_index:
    E.g. noise_index=-1, [13 -1 -1 13 -1 13 13 42 -1 -1] -> [1 0 0 1 0 1 1 2 0 0]

    If noise_index is not None but the input does not contain noise_index, 0
    will still be reserved for it:
    E.g. noise_index=-1, [13 4 4 13 4 13 13 42 4 4] -> [1 2 2 1 2 1 1 3 2 2]
    """
    unique_indices, locations = np.unique(input, return_inverse=True)
    cluster_index_map = np.arange(unique_indices.shape[0])
    if noise_index is not None:
        if noise_index in unique_indices:
            # Sort so that 0 aligns with the noise_index
            cluster_index_map = cluster_index_map[(unique_indices != noise_index).argsort()]
        else:
            # Still reserve 0 for noise, even if it's not present
            cluster_index_map += 1
    return np.take(cluster_index_map, locations)





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
            osp.basename(rootfile).replace(
                '.root',
                f'_{event.metadata["i_event"]:03d}_{event.metadata["endcap"]}.npz'
                )
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
        y = event.rechits.get("RecHitHGC_incrClusterIdx")
        n_noise = (y==0).sum()
        n_clusters = len(np.unique(y)) - 1 # don't count the noise cluster
        print(
            f'<Event'
            f'\n  metadata={format_and_indent(event.metadata)}'
            f'\n  rechits:'
            f'\n    shape={event.rechits.array.shape}'
            f'\n    keys={format_and_indent(event.rechits.keys)}'
            f'\n    n_noise={n_noise}'
            f'\n    n_clusters (non-noise)={n_clusters}'
            f'\n  simtracks:'
            f'\n    shape={event.simtracks.array.shape}'
            f'\n    keys={format_and_indent(event.simtracks.keys)}'
            f'\n  simclusters:'
            f'\n    shape={event.simclusters.array.shape}'
            f'\n    keys={format_and_indent(event.simclusters.keys)}'
            )
 