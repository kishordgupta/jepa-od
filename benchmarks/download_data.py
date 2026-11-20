"""Download the benchmark datasets used by bench_tabular.py and bench_ts.py.

The datasets are third-party and are not stored in this repository:

* ADBench classical datasets (Han et al., 2022): https://github.com/Minqi824/ADBench
* NAB realKnownCause (Numenta Anomaly Benchmark): https://github.com/numenta/NAB
* SKAB valve1 (Skoltech Anomaly Benchmark): https://github.com/waico/SKAB

Usage: python benchmarks/download_data.py [--dest data]
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
    dest = ap.parse_args().dest

    print('ADBench')
    for name in ADBENCH_FILES:
        fetch(ADBENCH.format(name), os.path.join(dest, 'adbench', f'{name}.npz'))
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
