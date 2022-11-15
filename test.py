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
