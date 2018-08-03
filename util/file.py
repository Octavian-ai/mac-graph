
import os.path
import sys
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

try:
  from google.cloud import storage
  import google.cloud.exceptions
except ImportError as e:
  logger.warn("Could not import google.cloud, will not save to bucket")
  pass


class FileThingy(object):

  def __init__(self, args, filename):
    self.args = args
    self.filename = filename

  @property
  def file_dir(self):
    return os.path.join(self.args.output_dir, self.args.run)

  @property
  def file_path(self):
    return os.path.join(self.args.output_dir, self.args.run, self.filename)

  @property
  def gcs_path(self):
    return os.path.join(self.args.gcs_dir, self.args.run, self.filename)
  

# TODO: See if TF GFile can replace this
def path_exists(path):
  return tf.gfile.Exists(path)
  

class FileReadie(FileThingy):
  """Tries to write on traditional filesystem and Google Cloud storage"""

  def __init__(self, args, filename, binary=False):
    super().__init__(args, filename)
    self.trad_file = None
    self.binary = binary

  def copy_from_bucket(self):
    if 'google.cloud' in sys.modules and self.args.bucket is not None and self.args.gcs_dir is not None:
      client = storage.Client()
      bucket = client.get_bucket(self.args.bucket)
      blob = bucket.blob(self.gcs_path)
      os.makedirs(self.file_dir, exist_ok=True)
      with open(self.file_path, "wb" if self.binary else "w") as dest_file:
        try:
          blob.download_to_file(dest_file)
        except google.cloud.exceptions.NotFound:
          raise FileNotFoundError()

  def __enter__(self):
    self.copy_from_bucket()
    self.trad_file = open(self.file_path, "rb" if self.binary else "r" )
    return self.trad_file

  def __exit__(self, type, value, traceback):
    self.trad_file.close()

    


class FileWritey(FileThingy):
  """Tries to write on traditional filesystem and Google Cloud storage"""

  def __init__(self, args, filename, binary=False):
    super().__init__(args, filename)
    self.trad_file = None
    self.open_str = "wb" if binary else "w" 

  def copy_to_bucket(self):
    if 'google.cloud' in sys.modules and self.args.bucket is not None and self.args.gcs_dir is not None:
      client = storage.Client()
      bucket = client.get_bucket(self.args.bucket)
      blob = bucket.blob(self.gcs_path)
      blob.upload_from_filename(filename=self.file_path)

  def __enter__(self):
    os.makedirs(self.file_dir, exist_ok=True)
    self.trad_file = open(self.file_path, self.open_str)
    return self.trad_file

  def __exit__(self, type, value, traceback):
    self.trad_file.close()
    self.copy_to_bucket()

    

