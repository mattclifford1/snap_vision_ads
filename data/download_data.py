import requests, zipfile, io, os

zip_file_url = 'https://drive.google.com/u/0/uc?id=1O-492BRWL-_7qN6hL2S4A4BSrLTnoQyw&export=download&confirm=t'

print('Downloading data')
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
if os.path.isdir('data'):
    extraction_path = 'data'
elif os.path.isdir('../data'):
    extraction_path = '../data'
else:
    os.mkdir('data')
    extraction_path = 'data'
print('Unzipping data')
z.extractall(extraction_path)
