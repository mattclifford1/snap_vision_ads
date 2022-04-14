import wget, zipfile, io, os


def download_zipped_model(zip_file_url):
    print('Downloading model')
    zip_file = wget.download(zip_file_url, bar=wget.bar_thermometer)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)


if __name__ == '__main__':
    url = 'https://drive.google.com/u/0/uc?id=1--XVjm3VGkMg450MC3jG3qGGCpHcP_ct&export=download&confirm=t'
    # url = 'https://drive.google.com/file/d/1--XVjm3VGkMg450MC3jG3qGGCpHcP_ct/view?usp=sharing'
    download_zipped_model(url)
