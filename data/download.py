import wget, zipfile, io, os


def _download_and_unzip(extraction_dir='data',
                        zip_file_url='https://drive.google.com/u/0/uc?id=1O-492BRWL-_7qN6hL2S4A4BSrLTnoQyw&export=download&confirm=t'):
    # download
    zip_file = wget.download(zip_file_url, bar=wget.bar_thermometer)
    if not os.path.isdir(extraction_dir):
        os.mkdir(extraction_dir)

    # unzip
    print('\nUnzipping into directory: ', extraction_dir)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)

    # delete zip file
    os.remove(zip_file)

    # remove mac os files
    DS_file = os.path.join(extraction_dir, '.DS_Store')
    if os.path.isfile(DS_file):
        os.remove(DS_file)


def get_if_doesnt_exist(dir):
    data_set = os.path.join(dir, 'uob_image_set')
    if not os.path.isdir(data_set):
        print('Downloading data')
        _download_and_unzip(dir)
        print('finished')
    return data_set


if __name__ == '__main__':
    if os.path.isdir('data'):
        extraction_path = 'data'
    elif os.path.isdir('../data'):
        extraction_path = '../data'
    get_if_doesnt_exist(extraction_path)
