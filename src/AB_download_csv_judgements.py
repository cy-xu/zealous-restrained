""" Use all results from contributors to merge the matching boxes
and outlier boxes, so the expert can review the results.
"""

import os
import json
import pandas as pd

import urllib.request

if __name__ == "__main__":
    # api_key = sys.argv[1]
    api_key = "AXZngREC8oBJS-Hy14Wy"

    base_dir = '/home/cyxu/hdd/ego4d_eval/AB_testing_results/'
    part1_dir = base_dir + 'part_1_merged_judgements/'
    part2_dir = base_dir + 'part_2_merged_judgements/'

    reviewed_dir = part2_dir + 'reviewed_judgements_v3/'
    os.makedirs(reviewed_dir, exist_ok=True)

    reviewed_csv = reviewed_dir + 'f1970104.csv'
    df_rw = pd.read_csv(reviewed_csv)

    # batch download reviewed judgements
    for i, row in df_rw.iterrows():
        clip = row['annotation'].split('/')[-1]
        clip = clip.replace('_merged.json', '_reviewed.json')
        # clip = row['Clip_name'] + f"_{row['_worker_id']}.json"

        annotation_pr = row['annotation_pr'].split('"')
        json_url = [s for s in annotation_pr if "https://" in s][0]
        json_url = json_url.replace("requestor-proxy", "api-beta")
        json_url += "&key=" + api_key

        with urllib.request.urlopen(json_url) as url:
            data = json.loads(url.read().decode())
            out_path = os.path.join(reviewed_dir, clip)

            with open(out_path, 'w') as outfile:
                json.dump(data, outfile)
                print(f'{clip} downloaded')