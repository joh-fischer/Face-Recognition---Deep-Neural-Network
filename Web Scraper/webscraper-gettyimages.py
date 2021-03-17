import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_images(url):
    soup = bs(requests.get(url).content, "html.parser")

    urls = []
    for img in tqdm(soup.find_all("img"), "Extracting images"):
        img_url = img.attrs.get("src")
        if not img_url:
            # if img does not contain src attribute, just skip
            continue

         # make the URL absolute by joining domain with the URL that is just extracted
        img_url = urljoin(url, img_url)

        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass


        # finally, if the url is valid
        if is_valid(img_url):
            urls.append(img_url)
    return urls

def download(url, pathname):
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))
    # get the file name
    filename = os.path.join(pathname, url.split("/")[-1] + '.jpg')
    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))

def main(url, path):
    # get all images
    imgs = get_all_images(url)
    for img in imgs:
        # for each image, download it
        download(img, path)


# path
dir_path = os.path.dirname(os.path.realpath(__file__))

# path to identities text file
identPath = os.path.join(dir_path, 'identities.txt')
ident_file = open(identPath, 'r')
lines = ident_file.readlines()

identities = []
id = 1
for line in lines:
    # get celeb name
    name = line.rstrip('\n')
    # get different search paths for the url
    searchPartA = name.replace(" ", "-")
    searchPartB = name.replace(" ", "%20")
    
    searchUrl = 'https://www.gettyimages.de/fotos/' + searchPartA + '?family=editorial&numberofpeople=one&phrase=' + searchPartB + '&sort=mostpopular#license'
    identities.append( (id, name, searchUrl) )
    id +=1

#urlToScrape = 'https://www.gettyimages.de/fotos/channing-tatum?family=editorial&numberofpeople=one&phrase=Channing%20tatum&sort=mostpopular#license'

storagePath = os.path.join(dir_path, 'SCRAPED')
for identity in identities:
    id, name, url = identity
    
    idPath = os.path.join(storagePath, str(id))

    main(url, idPath)