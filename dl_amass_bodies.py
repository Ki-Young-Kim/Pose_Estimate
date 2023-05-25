import requests
import re
import json
import os
import shutil
import time
from urllib.parse import unquote
import traceback
import functools

import requests
from bs4 import BeautifulSoup


# Download datasets from *.is.tue.mpg.de
# Save the value of PHPSESSID for the domain in ./sessionid.txt
DOMAIN = "amass.is.tue.mpg.de"


def load_sess():
    sess_id_path = "./sessionid.txt"
    
    with open(sess_id_path) as f:
        sess_id = f.read().strip()

    sess = requests.session()
    h = sess.get("https://" + DOMAIN)
    for ck in h.cookies:
        if ck.name == "PHPSESSID":
            ck.value = sess_id
    else:
        sess.cookies['PHPSESSID'] = sess_id
        for ck in sess.cookies:
            if ck.name == "PHPSESSID":
                ck.domain = DOMAIN

    return sess


def download_file(sess, url, fp, headers=None):
    with sess.get(url, stream=True, headers=headers) as resp:
        resp.raw.read = functools.partial(resp.raw.read, decode_content=True)
        with open(fp, 'wb') as f:
            shutil.copyfileobj(resp.raw, f)


def save_datasets_links():
    sess = load_sess()
    resp = sess.get("https://" + DOMAIN + "/download.php")
    soup = BeautifulSoup(resp.content, 'html.parser')

    datasets_all = []
    for tr in soup.select('tr'):
        btns = list(tr.select('button'))
        if len(btns) > 0:
            info_a = tr.select_one("td a")
            name = info_a.text
            name_long = info_a['title']
            datasets = []
            for btn in btns:
                ds_type = btn.text
                m = re.match(r"openModalLicense\('(.+)', (.+)\)", btn['onclick'])
                dl_url = unquote(m.group(1))
                licensename = m.group(2)
                filename = dl_url[dl_url.index("sfile=") + 6:]
                datasets.append({'type': ds_type, 'url': dl_url,
                    'licensename': licensename, 'filename': filename})
    
            datasets_all.append({'short_name': name, 'long_name': name_long, 'datasets': datasets})
    
    with open('datasets.json', 'w') as f:
        json.dump({'datasets': datasets_all}, f, indent=4)
          

def download_datasets(all_datasets_info):
    save_base_dir = 'amass_datasets'
    os.makedirs(save_base_dir, exist_ok=True)

    for i_dsets, dsets in enumerate(all_datasets_info):
        print("[{}/{}] Datasets \"{}\"".format(i_dsets+1, len(all_datasets_info), dsets['long_name'] or dsets['short_name']))
        dsets_base_dir = os.path.join(save_base_dir, dsets['short_name'])
        for ds in dsets['datasets']:
            dset_dir = os.path.join(dsets_base_dir, ds['type'].replace('+', 'p'))
            os.makedirs(dset_dir, exist_ok=True)
            dset_fn = os.path.basename(ds['url'])
            dset_fp = os.path.join(dset_dir, dset_fn)
            if os.path.isfile(dset_fp):
                print("{} already exists; skipping...".format(dset_fp))
            else:
                n_retries = 0
                while 1:
                    try:
                        sess = load_sess()

                        resp = sess.get(ds['url'])  # To set cookie for download site

                        license_accept_url = "https://amass.is.tue.mpg.de/admin/ajax_setlicenseagreed.php?filename="+ ds['filename'] + "&licensename=" + ds['licensename']
                        resp = sess.get(license_accept_url)
                        
                        download_file(sess, ds['url'], dset_fp,
                                headers={'Referer': 'https://' + DOMAIN + '/'})
                        break
                    except KeyboardInterrupt:
                        shutil.rmtree(dset_dir)
                        exit(1)
                    except:
                        shutil.rmtree(dset_dir)
                        traceback.print_exc()
                        wait_secs = 30 * n_retries
                        print("Download failed for {}; retrying in {} secs...".format(
                            ds['url'], wait_secs))
                        n_retries += 1
                        time.sleep(wait_secs)


if __name__ == '__main__':
    datasets_json_path = "datasets.json"
    if not os.path.isfile(datasets_json_path):
        print("Downloading datasets info...")
        save_datasets_links()

    print("Downloading datasets...")
    with open(datasets_json_path) as f:
        download_datasets(json.load(f)['datasets'])


