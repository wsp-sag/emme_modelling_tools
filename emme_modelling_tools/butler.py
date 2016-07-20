from __future__ import division

import os
from os import path
import pandas as pd
import sqlite3 as sqlite
from datetime import datetime as dt
from warnings import warn

from ..matrix_converters.matrix_converters.emme import from_emx, to_emx
from ..matrix_converters.matrix_converters.fortran import from_binary_matrix, to_binary_matrix
from ..matrix_converters.matrix_converters.common import expand_array, coerce_matrix

try:
    import inro.modeller as m
    mm = m.Modeller()
    project_emmebank = mm.emmebank
    del mm, m
except (ImportError, AssertionError):
    # AssertionError is thrown by Emme if a Modeller connection already exists.
    project_emmebank = None


class ButlerOverwriteWarning(RuntimeWarning):
    pass


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

    MATRIX_EXTENSION = 'bin'
    SUBDIRECTORY_NAME = 'emmebin'

    @staticmethod
    def create(parent_directory, zone_system, fortran_max_zones):
        """
        Creates a new (or clears and initializes and existing) MatrixButler.

        Args:
            parent_directory (unicode): The parent directory in which to keep the Butler.
            zone_system (pandas.Int64Index or List[int]): The zone system to conform to.
            fortran_max_zones (int): The total number of zones expected by the FORTRAN matrix reader.

        Returns:
            MatrixButler instance.
        """
        zone_system = pd.Int64Index(zone_system)
        fortran_max_zones = int(fortran_max_zones)

        butler_path = path.join(parent_directory, MatrixButler.SUBDIRECTORY_NAME)

        if not path.exists(butler_path):
            os.makedirs(butler_path)

        dbfile = path.join(butler_path, "matrix_directory.sqlite")
        db_exists = path.exists(dbfile)  # Connecting to a non-existent file will create the file, so cache this first
        db = sqlite.connect(dbfile)
        db.row_factory = sqlite.Row

        if db_exists:
            fortran_max_zones_existing, zone_system_existing = MatrixButler._preload(db)
            existing_is_compatible = fortran_max_zones == fortran_max_zones_existing and zone_system.equals(zone_system_existing)
            if not existing_is_compatible:
                msg = "Existing matrix cache not compatible with current zone system and will be cleared of any" \
                      " stored matrix files. Cache directory is '%s'" % parent_directory
                warn(ButlerOverwriteWarning(msg))

                for fn in os.listdir(butler_path):
                    if fn.endswith(MatrixButler.MATRIX_EXTENSION):
                        fp = path.join(butler_path, fn)
                        os.remove(fp)
                MatrixButler._clear_tables(db)
                MatrixButler._create_tables(db, zone_system, fortran_max_zones)
            # When the db exits AND is compatible, there's no need to create the tables
        else:
            MatrixButler._create_tables(db, zone_system, fortran_max_zones)

        return MatrixButler(butler_path, db, zone_system, fortran_max_zones)

    @staticmethod
    def _clear_tables(db):
        db.execute("DROP TABLE IF EXISTS properties;")
        db.execute("DROP TABLE IF EXISTS zone_system;")
        db.execute("DROP TABLE IF EXISTS matrices;")
        db.commit()

    @staticmethod
    def _create_tables(db, zone_system, fortran_max_zones):
        sql = """
        CREATE TABLE properties(
        name VARCHAR NOT NULL PRIMARY KEY,
        value VARCHAR
        );
        """
        db.execute(sql)

        sql = """
        INSERT INTO properties
        VALUES (?, ?)
        """
        db.execute(sql, ('max_zones_fortran', fortran_max_zones))

        sql = """
        CREATE TABLE zone_system(
        number INT NOT NULL PRIMARY KEY,
        zone INT
        );
        """
        db.execute(sql)

        sql = """
        INSERT INTO zone_system
        VALUES (?, ?)
        """
        for i, zone in enumerate(zone_system):
            zone = int(zone)  # Cast to Python INT, because SQLite doesn't like NumPy integers.
            db.execute(sql, (i, zone))

        sql = """
        CREATE TABLE matrices(
        id VARCHAR NOT NULL PRIMARY KEY,
        number INT,
        description VARCHAR,
        timestamp VARCHAR
        );
        """
        db.execute(sql)
        db.commit()

    @staticmethod
    def connect(parent_directory):
        """
        Connect to an existing MatrixButler, without initializing it

        Args:
            parent_directory (unicode): The parent directory in which to find the MatrixButler.

        Returns:
            IOError if a MatrixButler cannot be found at the given parent directory.

        """
        butler_path = path.join(parent_directory, MatrixButler.SUBDIRECTORY_NAME)
        if not os.path.exists(butler_path):
            raise IOError("No matrix butler found at '%s'" % parent_directory)

        dbfile = path.join(butler_path, "matrix_directory.sqlite")
        if not os.path.exists(dbfile):
            raise IOError("No matrix butler found at '%s'" % parent_directory)

        db = sqlite.connect(dbfile)
        db.row_factory = sqlite.Row
        fortran_max_zones, zone_system = MatrixButler._preload(db)

        return MatrixButler(butler_path, db, zone_system, fortran_max_zones)

    @staticmethod
    def _preload(db):
        sql = """
        SELECT *
        FROM properties
        WHERE name="max_zones_fortran"
        """
        result = list(db.execute(sql))
        fortran_max_zones = int(result[0]['value'])

        sql = """
        SELECT *
        FROM zone_system
        """
        result = list(db.execute(sql))
        zone_system = pd.Int64Index([int(record['zone']) for record in result])

        return fortran_max_zones, zone_system

    def __init__(self, *args):
        butler_path, db, zone_system, fortran_max_zones = args
        self._path = butler_path
        self._connection = db
        self._zone_system = zone_system
        self._max_zones_fortran = fortran_max_zones

        self._committing = True

    def _write_matrix_record(self, unique_id, number, description, timestamp):
        sql = """
        INSERT OR REPLACE INTO matrices
        VALUES (?, ?, ?, ?)
        """
        self._connection.execute(sql, (unique_id, number, description, timestamp))
        if self._committing:
            self._connection.commit()

    def _lookup_matrix(self, unique_id):
        sql = """
        SELECT *
        FROM matrices
        WHERE id=?
        """
        result = list(self._connection.execute(sql, [unique_id]))
        if not result:
            raise KeyError(unique_id)
        assert len(result) == 1
        mfn = result[0]['number']
        return "mf%s.%s" % (mfn, self.MATRIX_EXTENSION)

    def _next_mfid(self):
        """Gets the next available matrix ID from the current path."""
        i = 1
        fn = "mf%s.%s" % (i, self.MATRIX_EXTENSION)
        while os.path.exists(os.path.join(self._path, fn)):
            i += 1
            fn = "mf%s.%s" % (i, self.MATRIX_EXTENSION)
        return fn

    def _store_matrix(self, array, target_mfid):
        fp = path.join(self._path, target_mfid)

        n, _ = array.shape
        padding = self._max_zones_fortran - n
        if padding > 0:
            array = expand_array(array, padding)
        to_binary_matrix(array, fp)

    def _dispense_matrix(self, source_mfid):
        fp = path.join(self._path, source_mfid)  #'.'.join([source_mfid, self.MATRIX_EXTENSION]))
        return from_binary_matrix(fp, self._zone_system)

    def _copy_from_bank(self, source_mfid, target_mfid, emmebank):
        """Low-level function to get a matrix from Emmebank"""
        emmebank = project_emmebank if emmebank is None else emmebank
        assert emmebank is not None

        source_mfid = emmebank.matrix(source_mfid).id
        bank_path = os.path.dirname(emmebank.path)
        source_fp = os.path.join(bank_path, 'emmemat', source_mfid + ".emx")

        matrix = from_emx(source_fp, zones=self._zone_system).values
        self._store_matrix(matrix, target_mfid)

    def _copy_to_bank(self, source_mfid, target_mfid, emmebank):
        emmebank = project_emmebank if emmebank is None else emmebank
        assert emmebank is not None

        matrix = self._dispense_matrix(source_mfid).values
        target_mfid = emmebank.matrix(target_mfid).id

        bank_path = os.path.dirname(emmebank.path)
        target_fp = os.path.join(bank_path, 'emmemat', target_mfid + ".emx")
        to_emx(matrix, target_fp, emmebank.dimensions['centroids'])

    def save_matrix(self, dataframe_or_mfid, unique_id, description="", emmebank=None):
        """
        Passes a matrix to the butler for safekeeping.

        Args:
            dataframe_or_mfid (DataFrame or basestring): Specifies the matrix to save. If basestring, it is assumed to
                refer to a matrix in an Emmebank (see `emmebank`). Otherwise, a square DataFrame is required.
            unique_id (basestring): The unique identifier for this matrix.
            description (basestring): A brief description of the matrix.
            emmebank (Emmebank or None): If using an mfid for the first arg, its matrix will be pulled from this
                Emmebank. Defaults to the Emmebank of the current Emme project when launched from Emme Python
        """
        try:
            target_mfid = self._lookup_matrix(unique_id)
        except KeyError:
            target_mfid = self._next_mfid()

        if isinstance(dataframe_or_mfid, basestring):
            self._copy_from_bank(dataframe_or_mfid, target_mfid, emmebank)
        elif isinstance(dataframe_or_mfid, (pd.DataFrame, pd.Series)):
            matrix = coerce_matrix(dataframe_or_mfid, allow_raw=True)
            self._store_matrix(matrix, target_mfid)
        else:
            raise TypeError(type(dataframe_or_mfid))

        target_number = target_mfid.split('.')[0][2:]
        self._write_matrix_record(unique_id, target_number, description, str(dt.now()))

    def load_matrix(self, unique_id, target_mfid=None, emmebank=None):
        """
        Gets a matrix from the butler.

        Args:
            unique_id (unicode): The name you gave to the butler for safekeeping.
            target_mfid (unicode or None): If provided, the butler will copy the matrix into the Emmebank at this
                given matrix ID or name. This matrix must already exist
            emmebank (Emmebank or None): Alternate Emmebank in which to save the matrix, if `target_mfid` is provided.
                Defaults to the Emmebank of the current Emme project when launched from Emme Python

        Returns: DataFrame or None, depending on whether `target_mfid` is given.

        Raises:
            KeyError if unique_id is not in the butler.
            AttributeError if target_mfid does not exist.

        """
        source_mfid = self._lookup_matrix(unique_id)

        if target_mfid is not None:
            self._copy_to_bank(source_mfid, target_mfid, emmebank)
        else:
            return self._dispense_matrix(source_mfid)

    def init_matrix(self, unique_id, description=""):
        """
        Registers a new (or zeros an old) matrix with the butler.

        Args:
            unique_id (unicode): The unique identifier for this matrix.
            description (unicode):  A brief description of the matrix.

        """
        zero_matrix = pd.DataFrame(0.0, index=self._zone_system, columns=self._zone_system)
        self.save_matrix(zero_matrix, unique_id, description)

    def delete_matrix(self, unique_id):
        """
        Deletes a matrix from the butler's directory

        Args:
            unique_id (unicode): The unique identifier of the matrix to delete.

        Raises:
            KeyError if unique_id cannot be found.

        """
        fn = self._lookup_matrix(unique_id)
        fp = path.join(self._path, fn)
        os.remove(fp)

        sql = """
        DELETE FROM matrices
        WHERE id=?
        """
        self._connection.execute(sql, [unique_id])
        if self._committing:
            self._connection.commit()

    def __del__(self):
        self._connection.commit()
        self._connection.close()

    '''
    When used as a context manager, matrix writes can be 'batched'; i.e. changes to the DB won't be committed until
    the context is closed. Committing to the DB is quite expensive (it doubles the run time when writing a new matrix)
    so this should save some time when the user wants to batch-out several matrices.
    '''

    def __enter__(self):
        self._committing = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.commit()
        self._committing = True
