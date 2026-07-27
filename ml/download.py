import requests, tarfile, io
from tqdm import tqdm
from helpers import DATA_DIR

# sources
OPENSCORE_URL = 'https://codeload.github.com/OpenScore/Lieder/tar.gz/refs/heads/main'
PDMX_RECORD = 'https://zenodo.org/api/records/14648209'
PDMX_SUBSET = 'rated_deduplicated' # 13187 of PDMX's 254077 scores: the ones users rated, near duplicates dropped


def zenodo_url(filename):
    files = requests.get(PDMX_RECORD, timeout=30).json()['files']
    return next(f['links']['self'] for f in files if f['key'] == filename)


def extract_mxl(url, dest, wanted=None):
    # both corpora ship as a single .tar.gz, so unpack it straight out of the download: members
    # arrive one at a time, so each score is written as it goes past and the archive itself (1.9 GB
    # for PDMX) never has to land on disk. The archive's own root folder is dropped and what's left
    # is flattened, so a dataset stays one directory of uniquely named scores
    dest.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode='r|gz') as tar:
            for member in tqdm(tar, f'Downloading {dest.name}...'):
                name = member.name.split('/', 1)[-1]
                if not name.endswith('.mxl'): continue
                if wanted is not None and name.rsplit('/', 1)[-1] not in wanted: continue
                (dest / name.replace('/', '_')).write_bytes(tar.extractfile(member).read())


def download_openscore():
    extract_mxl(OPENSCORE_URL, DATA_DIR / 'openscore')


def download_pdmx():
    # PDMX is ~250k scores of every quality, but its subset lists come as their own small archive,
    # so read the subset we want first and pull only those scores out of the big one
    subsets = requests.get(zenodo_url('subset_paths.tar.gz'), timeout=60).content
    with tarfile.open(fileobj=io.BytesIO(subsets)) as tar:
        paths = tar.extractfile(f'subset_paths/{PDMX_SUBSET}.txt').read().decode().split()
    # the lists point at each score's .json metadata, which is named after the .mxl beside it
    wanted = { path.rsplit('/', 1)[-1].replace('.json', '.mxl') for path in paths }
    extract_mxl(zenodo_url('mxl.tar.gz'), DATA_DIR / 'pdmx', wanted)


DATASETS = { 'openscore': download_openscore, 'pdmx': download_pdmx }


def download_datasets():
    for name, download in DATASETS.items():
        scores = list((DATA_DIR / name).glob('*.mxl'))
        if scores:
            print(f'{name}: {len(scores)} scores already downloaded')
        else:
            download()


if __name__ == "__main__":
    download_datasets()
