import os.path as osp
import seutils, uproot, numpy as np
import hgcal_npz



test_file = osp.join(osp.dirname(osp.abspath(__file__)), 'seed0_n100.root')

def download_test_file():
    if not osp.isfile(test_file):
        src = 'root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/hgcal_npz/seed0_n100.root',
        hgcal_npz.logger.info(f'Copying test file {src} -> {test_file}')
        seutils.cp(src, test_file)

download_test_file()


def test_events_factory():
    for event in hgcal_npz.events_factory(test_file):
        break

    # Load some braches manually to 
    t = uproot.open(test_file + ':Events')
    hlarray = t.arrays(['RecHitHGC_x', 'SimTrack_pdgId', 'MergedSimCluster_pdgId'])
    rechit_x = hlarray['RecHitHGC_x'][0].to_numpy()
    simtrack_pdgid = hlarray['SimTrack_pdgId'][0].to_numpy()
    simcluster_pdgid = hlarray['MergedSimCluster_pdgId'][0].to_numpy()

    assert len(event.rechits) == rechit_x.shape[0]
    assert len(event.rechits.keys) == event.rechits.array.shape[1]
    assert len(event.simtracks) == len(simtrack_pdgid)
    assert len(event.simtracks.keys) == event.simtracks.array.shape[1]
    assert len(event.simclusters) == len(simcluster_pdgid)
    assert len(event.simclusters.keys) == event.simclusters.array.shape[1]

    np.testing.assert_array_equal(rechit_x, event.rechits.to_numpy(['RecHitHGC_x']).flatten())
    np.testing.assert_array_equal(simcluster_pdgid, event.simclusters.to_numpy(['MergedSimCluster_pdgId']).flatten())


def test_io():
    for event in hgcal_npz.events_factory(test_file):
        break

    event.save('test.npz')
    other = hgcal_npz.Event.load('test.npz')

    assert event.rechits == other.rechits
    assert event.simtracks == other.simtracks
    assert event.simclusters == other.simclusters

    other.metadata.pop('npz_file', None)
    assert event.metadata == other.metadata


def test_split_and_flip():
    event = hgcal_npz.Event()
    event.rechits = hgcal_npz.DataFrame(['hitvar_x', 'hitvar_z', 'hitvar_eta'])
    event.rechits.array = np.array([
        [2., -2., -2.],
        [1., -1., -1.],
        [1., 1., 1.],
        [2., 2., 2.],
        ])
    event.simtracks = hgcal_npz.DataFrame(['trackvar_x', 'trackvar_z', 'trackvar_eta'])
    event.simtracks.array = np.array([
        [2., -2., -2.],
        [1., -1., -1.],
        [1., 1., 1.],
        [2., 2., 2.],
        ])
    event.simclusters = hgcal_npz.DataFrame(['clusvar_x', 'clusvar_z', 'clusvar_eta'])
    event.simclusters.array = np.array([
        [-2., -2., -2.],
        [1., -1., -1.],
        [1., 1., 1.],
        [2., 2., 2.],
        ])

    pos, neg = hgcal_npz.split(event, ['hitvar_z', 'trackvar_z', 'clusvar_z'])

    np.testing.assert_array_equal(
        pos.rechits.array,
        np.array([
            [1., 1., 1.],
            [2., 2., 2.],
            ])
        )
    np.testing.assert_array_equal(
        neg.rechits.array,
        np.array([
            [2., -2., -2.],
            [1., -1., -1.],
            ])
        )
    np.testing.assert_array_equal(
        neg.simclusters.array,
        np.array([
            [-2., -2., -2.],
            [1., -1., -1.],
            ])
        )

    hgcal_npz.flip(
        neg,
        ['hitvar_z', 'hitvar_eta', 'trackvar_z', 'trackvar_eta', 'clusvar_z', 'clusvar_eta']
        )

    np.testing.assert_array_equal(
        pos.rechits.array,
        np.array([
            [1., 1., 1.],
            [2., 2., 2.],
            ])
        )
    np.testing.assert_array_equal(
        neg.rechits.array,
        np.array([
            [2., 2., 2.],
            [1., 1., 1.],
            ])
        )
    np.testing.assert_array_equal(
        neg.simclusters.array,
        np.array([
            [-2., 2., 2.],
            [1., 1., 1.],
            ])
        )


def test_augment():
    for event in hgcal_npz.events_factory(test_file):
        break
    assert 'RecHitHGC_theta' in event.rechits.keys


def test_incremental_cluster_index():
    input = np.array([13, 4, 4, 13, 4, 13, 13, 42, 4, 4])
    np.testing.assert_array_equal(
        hgcal_npz.incremental_cluster_index(input),
        np.array([1, 0, 0, 1, 0, 1, 1, 2, 0, 0])
        )
    # Noise index should get 0 if it is supplied:
    np.testing.assert_array_equal(
        hgcal_npz.incremental_cluster_index(input, noise_index=13),
        np.array([0, 1, 1, 0, 1, 0, 0, 2, 1, 1])
        )
    # 0 should still be reserved for noise_index even if it is not present:
    np.testing.assert_array_equal(
        hgcal_npz.incremental_cluster_index(input, noise_index=-99),
        np.array([2, 1, 1, 2, 1, 2, 2, 3, 1, 1])
        )
