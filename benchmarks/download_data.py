"""Download the benchmark datasets used by bench_tabular.py and bench_ts.py.

The datasets are third-party and are not stored in this repository:

* ADBench classical datasets (Han et al., 2022): https://github.com/Minqi824/ADBench
* NAB realKnownCause (Numenta Anomaly Benchmark): https://github.com/numenta/NAB
* SKAB valve1 (Skoltech Anomaly Benchmark): https://github.com/waico/SKAB

Usage: python benchmarks/download_data.py [--dest data] [--all-adbench] [--tsb-published]

--all-adbench also fetches the other 31 ADBench classical datasets into
data/adbench_extra/ (about 72 MB) for benchmarks/bench_extended.py.

--tsb-published fetches only the per-file VUS-PR that TSB-AD published for its
reference detectors (https://github.com/TheDatumOrg/TSB-AD, Apache-2.0; two CSV
files, 0.3 MB) into data/tsb_published/ for benchmarks/make_tsb_tables.py. The
TSB-AD series themselves (TSB-AD-M.zip 540 MB, TSB-AD-U.zip 73 MB, from
https://www.thedatum.org/datasets/) are not fetched by this script; see
benchmarks/bench_tsb.py.
"""
import argparse
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ADBENCH = 'https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{}.npz'
ADBENCH_FILES = [
    '6_cardio', '12_fault', '17_InternetAds', '18_Ionosphere', '22_magic.gamma',
    '25_musk', '29_Pima', '31_satimage-2', '33_skin', '36_speech', '38_thyroid',
    '40_vowels', '41_Waveform', '42_WBC', '44_Wilt', '45_wine',
]

# the remaining ADBench classical datasets, used by bench_extended.py only
ADBENCH_EXTRA_FILES = [
    '1_ALOI', '2_annthyroid', '3_backdoor', '4_breastw', '5_campaign', '7_Cardiotocography',
    '8_celeba', '9_census', '10_cover', '11_donors', '13_fraud', '14_glass', '15_Hepatitis',
    '16_http', '19_landsat', '20_letter', '21_Lymphography', '23_mammography', '24_mnist',
    '26_optdigits', '27_PageBlocks', '28_pendigits', '30_satellite', '32_shuttle', '34_smtp',
    '35_SpamBase', '37_Stamps', '39_vertebral', '43_WDBC', '46_WPBC', '47_yeast',
]

NAB = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/{}.csv'
NAB_LABELS = 'https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json'
NAB_FILES = [
    'ambient_temperature_system_failure', 'cpu_utilization_asg_misconfiguration',
    'ec2_request_latency_system_failure', 'machine_temperature_system_failure',
    'nyc_taxi', 'rogue_agent_key_hold', 'rogue_agent_key_updown',
]

SKAB = 'https://raw.githubusercontent.com/waico/SKAB/master/data/valve1/{}.csv'
SKAB_FREE = 'https://raw.githubusercontent.com/waico/SKAB/master/data/anomaly-free/anomaly-free.csv'
SKAB_FILES = list(range(16))


# the commit of TSB-AD whose metric code and file lists the benchmark used
TSB_PUBLISHED = ('https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/303dffbfc11612995b17b06633f8c9d0ea0e38e5/'
                 'benchmark_exp/benchmark_eval_results/{}_mergedTable_VUS-PR.csv')


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'  {url}')
    tmp = path + '.part'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--all-adbench', action='store_true',
                    help='also fetch the other 31 ADBench datasets into <dest>/adbench_extra')
    ap.add_argument('--tsb-published', action='store_true',
                    help="fetch only TSB-AD's published per-file VUS-PR into <dest>/tsb_published")
    args = ap.parse_args()
    dest = args.dest

    if args.tsb_published:
        for name in ('multi', 'uni'):
            fetch(TSB_PUBLISHED.format(name), os.path.join(dest, 'tsb_published', f'{name}_mergedTable_VUS-PR.csv'))
        print(f'done: {dest}')
        return
    print('ADBench')
    for name in ADBENCH_FILES:
        fetch(ADBENCH.format(name), os.path.join(dest, 'adbench', f'{name}.npz'))
    if args.all_adbench:
        print('ADBench (remaining datasets)')
        for name in ADBENCH_EXTRA_FILES:
            fetch(ADBENCH.format(name), os.path.join(dest, 'adbench_extra', f'{name}.npz'))
    print('NAB')
    fetch(NAB_LABELS, os.path.join(dest, 'nab', 'combined_windows.json'))
    for name in NAB_FILES:
        fetch(NAB.format(name), os.path.join(dest, 'nab', f'{name}.csv'))
    print('SKAB')
    fetch(SKAB_FREE, os.path.join(dest, 'skab', 'anomaly-free.csv'))
    for i in SKAB_FILES:
        fetch(SKAB.format(i), os.path.join(dest, 'skab', f'valve1_{i}.csv'))
    print(f'done: {dest}')


if __name__ == '__main__':
    main()
