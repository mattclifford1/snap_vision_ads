import wget, zipfile, io, os

# download
zip_file_url = 'https://drive.google.com/u/0/uc?id=1O-492BRWL-_7qN6hL2S4A4BSrLTnoQyw&export=download&confirm=t'
print('Downloading data')
r = wget.download(zip_file_url, bar=wget.bar_thermometer)

# get where to unzip contents to
if os.path.isdir('data'):
    extraction_path = 'data'
elif os.path.isdir('../data'):
    extraction_path = '../data'
else:
    os.mkdir('data')
    extraction_path = 'data'

# unzip
print('\nUnzipping data to: ', extraction_path)
zip_file = 'uob_image_set.zip'
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# delete zip file
os.remove(zip_file)

# remove mac os files
DS_file = os.path.join(extraction_path, '.DS_Store')
if os.path.isfile(DS_file):
    os.remove(DS_file)
