import requests
from tqdm import tqdm

def poorman_wget(url, filename):
    ckpt_request = requests.get(url, stream=True)
    request_status = ckpt_request.status_code

    total_size = int(ckpt_request.headers.get("Content-Length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError("You have not accepted the license for this model.")
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

    # write to model path
    with open(filename, 'wb') as model_file:
        for data in ckpt_request.iter_content(block_size):
            model_file.write(data)
            progress_bar.update(len(data))

    progress_bar.close()

def wget_headers(url):
    r = requests.get(url, stream=True, headers={'Connection':'close'})
    return r.headers

def wget_progress(url, filename, length=0, chunk_size=8192, callback=None):

    one_percent = int(length) / 100
    next_percent = 1

    with requests.get(url, stream=True) as r:

        r.raise_for_status()
        downloaded_bytes = 0
        callback(next_percent)
        with open(filename, 'wb') as f:
            try:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk:
                    f.write(chunk)
                    downloaded_bytes += chunk_size
                    if downloaded_bytes > next_percent * one_percent:
                        next_percent += 1
                        callback(next_percent)
            except Exception as e:
                print('error while writing download file: ', e)
