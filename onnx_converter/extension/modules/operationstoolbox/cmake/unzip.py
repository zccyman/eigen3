import argparse
import zipfile
import gzip


def decompress(file_name):
    try:
        f_name = file_name.split('.')[0]
        g_file = gzip.GzipFile(filename=file_name)
        open(f_name, "wb+").write(g_file.read())
        g_file.close()
    except:
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name',
                        type=str,
                        default='./build/ort_package/ort_lib.tar.gz',
                        help='library path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    decompress(args.file_name)
