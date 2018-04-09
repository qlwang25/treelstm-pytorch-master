"""
Downloads the following:
- Stanford parser
- Stanford POS tagger
- Glove vectors
- SICK dataset (semantic relatedness task)
"""

from __future__ import print_function
from urllib.request import urlopen
import sys
import os
from tqdm import tqdm
import shutil
import zipfile
import gzip
from six.moves.urllib.request import urlretrieve


def reporthook(t):
  """https://github.com/tqdm/tqdm"""
  last_b = [0]

  def inner(b=1, bsize=1, tsize=None):
    """
    b: int, optional
        Number of blocks just transferred [default: 1].
    bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
    tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
    """
    if tsize is not None:
        t.total = tsize
    t.update((b - last_b[0]) * bsize)
    last_b[0] = b
  return inner


def maybe_download(url, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename
    num_bytes=None disables the file size check."""
    local_filename = None
    filename = url.split('/')[-1]
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print("Downloading file {}...".format(url))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url, os.path.join(prefix, filename), reporthook=reporthook(t))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix, filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print("File {} successfully loaded".format(filename))
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename


def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_tagger(dirpath):
    tagger_dir = 'stanford-tagger'
    if os.path.exists(os.path.join(dirpath, tagger_dir)):
        print('Found Stanford POS Tagger - skip')
        return
    url = 'http://nlp.stanford.edu/software/stanford-postagger-2015-01-29.zip'
    filepath = maybe_download(url, dirpath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
        zf.close()
    os.remove(filepath)


def download_parser(dirpath):
    parser_dir = 'stanford-parser'
    if os.path.exists(os.path.join(dirpath, parser_dir)):
        print('Found Stanford Parser - skip')
        return
    url = 'http://nlp.stanford.edu/software/stanford-parser-full-2015-01-29.zip'
    filepath = maybe_download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, parser_dir))


def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(maybe_download(url, dirpath))


def download_sick(dirpath):
    if os.path.exists(dirpath):
        print('Found SICK dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    train_url = 'http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip'
    trial_url = 'http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip'
    test_url = 'http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip'
    unzip(maybe_download(train_url, dirpath))
    unzip(maybe_download(trial_url, dirpath))
    unzip(maybe_download(test_url, dirpath))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # data
    data_dir = os.path.join(base_dir, 'data')
    wordvec_dir = os.path.join(data_dir, 'glove')
    sick_dir = os.path.join(data_dir, 'sick')

    # libraries
    lib_dir = os.path.join(base_dir, 'lib')

    # download dependencies
    download_tagger(lib_dir)
    download_parser(lib_dir)
    download_wordvecs(wordvec_dir)
    download_sick(sick_dir)
