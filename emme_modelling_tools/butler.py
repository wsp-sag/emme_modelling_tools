import os
from os import path
import pandas as pd

from ..matrix_converters.matrix_converters.emme import from_emx, to_emx
from ..matrix_converters.matrix_converters.fortran import from_binary_matrix, to_binary_matrix

import inro.modeller as m
mm = m.Modeller()
project_emmebank = mm.emmebank


class MatrixButler(object):
    """
    Binary matrix file manager for working with Bill Davidson's FORTRAN code. He requires a LOT of matrices, and using
    this class keeps the Emmebank clean.

    A MatrixButler instance "connects" to a folder on the disk (or creates it if it doesn't exist). It mimics Emme's
    'emmemat' directory (which stores EMX files in the form 'mf1.emx, mf2.emx, mf3.emx...). Bill's code also expects
    matrices in this format.

    It should be noted that this version of MatrixButler reads and writes 'FORTRAN simple binary' files, rather than
    the 'EMX' format.
    """

    METADATA_SCHEMA = ['unique_id', 'mfid', 'description', 'timestamp']
    MATRIX_EXTENSION = 'bin'
    SUBDIRECTORY_NAME = 'emmebin'

    def __init__(self, parent_directory, zone_system, emme_max_zones, fortran_max_zones, allow_overwrite=False):
        """
        Constructs a new MatrixButler

        Args:
            parent_directory (unicode): Filepath to parent directory. The Butler's files will live in
                <parent_directory>/emmemat
            zone_system (List or ndarray): The zone system. Matrices given to the butler must be compatible with this
                zone system
            emme_max_zones (int): The maximum number of zones dimensioned by the current Emme license.
            fortran_max_zones (int):
            allow_overwrite (bool):
        """
        self._path = path.join(parent_directory, self.SUBDIRECTORY_NAME)
        self._config_path = path.join(self._path, 'config.ini')
        self._directory_path = path.join(self._path, 'matrix_directory.csv')

        try:
            self._read_config_file()
            self._read_matrix_directory()

            assert self._max_zones_emme == emme_max_zones
            assert self._max_zones_fortran == fortran_max_zones
            assert self._zone_system.equals(zone_system)
        except:
            if not allow_overwrite:
                raise

            # If any error is found (and overwriting is allowed), clobber the existing metadata and Butler directory
            # with the constructor args.

            if path.exists(self._path):
                # Wipe any existing binary matrices in the existing folder
                for fn in os.listdir(self._path):
                    if fn.endswith(self.MATRIX_EXTENSION):
                        fp = path.join(self._path, fn)
                        os.remove(fp)
            else:
                os.makedirs(self._path)
            self._metadata = pd.DataFrame(columns=self.METADATA_SCHEMA).set_index('unique_id')
            self._zone_system = pd.Index(zone_system)
            self._max_zones_emme = int(emme_max_zones)
            self._max_zones_fortran = int(fortran_max_zones)

    def _read_config_file(self):
        with open(self._config_path) as reader:
            self._max_zones_emme = int(reader.readline().strip())
            self._max_zones_fortran = int(reader.readline().strip())
            self._zone_system = pd.Index([int(i) for i in reader.readline().strip().split(',')])

    def _write_config_file(self):
        lines = [
            str(self._max_zones_emme), str(self._max_zones_fortran), ','.join(str(z) for z in self._zone_system)
        ]
        with open(self._config_path, 'w') as writer:
            writer.write('\n'.join(lines))

    def _read_matrix_directory(self):
        df = pd.read_csv(self._directory_path, index_col='unique_id', usecols=self.METADATA_SCHEMA)
        df.index.name = self.METADATA_SCHEMA[0]
        self._metadata = df

    def _write_matrix_directory(self):
        self._metadata.to_csv(self._directory_path, index=True, header=True)

    def __del__(self):
        self._write_config_file()
        self._write_matrix_directory()

    def save_matrix(self, dataframe_or_mfid, unique_id, description, emmebank=None):
        """
        Passes a matrix to the butler for safekeeping.

        Args:
            dataframe_or_mfid (DataFrame or basestring): Specifies the matrix to save. If basestring, it is assumed to
                refer to a matrix in an Emmebank (see `emmebank`). Otherwise, a square DataFrame is required.
            unique_id (basestring): The unique identifier for this matrix.
            description (basestring): A brief description of the matrix.
            emmebank (Emmebank or None): If using an mfid for the first arg, its matrix will be pulled from this
                Emmebank. If None, the current Emmebank will be used.
        """
        pass

    def _copy_from_bank(self, target_mfid, source_mfid, emmebank):
        pass

    def init_matrix(self, unique_id, description):
        """
        Registers a new (or zeros an old) matrix with the butler.

        Args:
            unique_id (unicode): The unique identifier for this matrix.
            description (unicode):  A brief description of the matrix.

        """
        pass

    def load_matrix(self, unique_id, target_mfid=None, emmebank=None):
        """
        Gets a matrix from the butler.

        Args:
            unique_id (unicode): The name you gave to the butler for safekeeping.
            target_mfid (unicode or None): If provided, the butler will copy the matrix into the Emmebank at this
                given matrix ID or name. This matrix must already exist
            emmebank (Emmebank or None): Alternate Emmebank in which to save the matrix, if `target_mfid` is provided.
                Defaults to the Emmebank of the current project

        Returns: DataFrame or None, depending on whether `target_mfid` is given.

        Raises:
            KeyError if unique_id is not in the butler.
            AttributeError if target_mfid does not exist.

        """
        pass
