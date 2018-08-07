
import os.path
import sys
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

# Now that I know tf.gfile is a thing, none of this is needed


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
  

def path_exists(path):
  return tf.gfile.Exists(path)
  

class FileReadie(FileThingy):
  """Tries to write on traditional filesystem and Google Cloud storage"""

  def __init__(self, args, filename, binary=False):
    super().__init__(args, filename)
    self.open_str = "rb" if binary else "r"

  def __enter__(self):
    self.file = tf.gfile.GFile(self.file_path, self.open_str)
    return self.file

  def __exit__(self, type, value, traceback):
    self.file.close()

    


class FileWritey(FileThingy):
  """Tries to write on traditional filesystem and Google Cloud storage"""

  def __init__(self, args, filename, binary=False):
    super().__init__(args, filename)
    self.open_str = "wb" if binary else "w" 

  def __enter__(self): 
    try:
      os.makedirs(self.file_dir, exist_ok=True)
    except Exception:
      pass

    self.file = tf.gfile.GFile(self.file_path, self.open_str)

    return self.file

  def __exit__(self, type, value, traceback):
    self.file.close()

    

