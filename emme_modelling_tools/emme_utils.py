from __future__ import division

from contextlib import contextmanager

from shapely.geometry import Point, LineString, Polygon
import shapelib as shp
import dbflib as dbf
from os import path
import os

import pandas as pd
import numpy as np

from inro.emme.matrix import MatrixData
from inro.emme.core.exception import ProtectionError, CapacityError
import inro.modeller as m
mm = m.Modeller()
project_emmebank = mm.emmebank


def init_matrix(mtx_id=None,  matrix_type="FULL", default=0.0, name="", description="", log=True):
    """
    "Safe" initialization of a matrix either creates (if new) or initializes a matrix in the Emmebank.

    Args:
        mtx_id (str or None): A valid matrix identifier, or None to get the next available new matrix. If None,
            `matrix_type` will be used.
        matrix_type (str): A valid matrix type, used only if `matrix_id` is None. As of Emme 4.2.4, valid options are
            'FULL", 'ORIGIN', 'DESTINATION', and 'SCALAR'
        default (float): The default value to initialize the matrix with.
        name (str): The name to give the matrix.
        description (str): The description to give the matrix
        log (bool): Set to False to disable reporting to Logbook.

    Returns:
        A Matrix instance representing the new or initialized matrix.

    """
    embank = mm.emmebank

    if mtx_id is None:
        mtx_id = embank.available_matrix_identifier(matrix_type)

    matrix = embank.matrix(mtx_id)

    if matrix is None:
        matrix = embank.create_matrix(mtx_id, default_value=default)
        if log:
            m.logbook_write("Created new %s = %s" % (mtx_id, default))
    else:
        if matrix.read_only:
            raise ProtectionError(mtx_id)

        matrix.initialize(default)
        if log:
            m.logbook_write("Initialized %s -> %s" % (mtx_id, default))

    matrix.name = name
    matrix.description = description

    return matrix


@contextmanager
def temporary_matrices(matrices=1, matrix_type='FULL', emmebank=None, log=False, id=False, squeeze=True):
    """
    Context manager to create multiple temporary matrices

    Args:
        matrix_type:
        matrices (int or Iterable): If integer, describes how many temporary matrices will be created. If an
            Iterable, each item will be taken as a matrix description.
        emmebank (Emmebank or None): The Emmebank object in which to create the matrices. Set to None to use the
            current Emmebank.
        log (bool): Set to True to record each temporary matrix creation to the Logbook.
        id (bool): Set to True to return a matrix ID. Otherwise a Matrix object is returned.
        squeeze (bool): Set to true to yield one temporary matrix instead of a list of length 1 when only one matrix
            is requested.

    Returns:
        basestring or Matrix, depending on the value of `id`.

    """
    emmebank = project_emmebank if emmebank is None else emmebank

    try:
        descriptors = list(matrices)
    except TypeError:
        descriptors = ["Temporary matrix %s" % i for i in xrange(matrices)]

    created_matrices = []
    try:
        for description in descriptors:
            matrix = init_matrix(description=description, log=log, matrix_type=matrix_type)
            if id: matrix = matrix.id
            created_matrices.append(matrix)

        if len(created_matrices) == 1 and squeeze:
            yield created_matrices[0]
        else:
            yield created_matrices
    finally:
        for matrix in created_matrices:
            mfid = matrix.id if not id else matrix
            emmebank.delete_matrix(mfid)


def init_extra_attribute(scenario, exatt_id, domain='LINK', default=0, description=''):
    """
    Initializes an extra attribute to a default value, creating it if it doesn't exist.

    Args:
        scenario:
        exatt_id:
        domain:
        default:
        description:

    Returns:

    """
    exatt = scenario.extra_attribute(exatt_id)
    if exatt is None:
        exatt = scenario.create_extra_attribute(domain, exatt_id, default)
    exatt.initialize(default)
    exatt.description = description


def next_available_scenario(emmebank):
    """
    Finds the next available scenario number for a given Emmebank.

    Args:
        emmebank (Emmebank instance): The Emmebank to query.

    Returns:
        int - The number of the next available scenario

    Raises:
        CapacityError if the Emmebanmk is full.

    """

    for i in range(1, emmebank.dimensions['scenarios'] + 1):
        if emmebank.scenario(i) is None: return i

    raise CapacityError("Not enough space in Emmebank for temporary scenario")


class ShapelyGeometry(object):
    """Composite Geometry + Dictionary"""

    def __init__(self, geom, attrs):
        self._geom = geom
        self._attrs = dict(attrs)

    def __getattr__(self, item):
        # If there's a name collision, we're a Geometry first and a Dictionary second.
        try:
            return getattr(self._geom, item)
        except AttributeError:
            return getattr(self._attrs, item)

    def __getitem__(self, item):
        return self._attrs[item]

    def __setitem__(self, item, value):
        self._attrs[item] = value

    def __dir__(self):
        return dir(self._geom) + dir(self._attrs)

    def __instancecheck__(self, instance):
        return isinstance(self._geom, instance) or isinstance(self._attrs, instance)


def _make_geom(record):
    shapetype = record.type
    if shapetype == shp.SHPT_POINT:
        return Point(record.vertices()[0])
    if shapetype == shp.SHPT_ARC:
        return LineString(record.vertices()[0])
    if shapetype == shp.SHPT_POLYGON:
        v = record.vertices()
        shell = v[0]
        holes = v[1:]
        return Polygon(shell, holes)
    raise RuntimeError("Unrecognized shape type %s" % shapetype)


def read_shapefile(fp, heal=True):
    """
    Reads an ESRI Shapeifle and converts it to Shapely Geometry objects. The DBF is also read and each returned geometry
    has its DBF attributes attached.

    Examples:
        points = read_shapefile("airports.shp")

        p0 = points[0]
        p0.geom_type
        >>'Point'
        p0['Airport_code']
        >>'YYZ'

    Args:
        fp (str or unicode): The path to the shapefile
        heal (bool): If true, this function will attempt to auto-heal any invalid the geometry using buffer(0).

    Returns:
        List of ShapelyGeometry objects

    """
    basepath, _ = path.splitext(fp)
    shp_path = basepath + ".shp"
    dbf_path = basepath + ".dbf"

    shapefile = shp.open(shp_path)
    try:
        shp_length, shp_type, x_extents, y_extents = shapefile.info()
        tablefile = dbf.open(dbf_path)
    except:
        shapefile.close()
        raise

    try:
        geoms = []
        assert shp_length == tablefile.record_count()
        for fid in xrange(shp_length):
            record = shapefile.read_object(fid)
            geom = _make_geom(record)
            if not geom.is_valid and heal:
                geom = geom.buffer(0)

            attrs = tablefile.read_record(fid)

            geoms.append(ShapelyGeometry(geom, attrs))
        return geoms
    finally:
        shapefile.close()
        tablefile.close()


def load_node_dataframe(scenario, pythonize_exatts=False):
    """
    Creates a table for node attributes in a scenario.

    Args:
        scenario: An instance of inro.emme.scenario.Scenario
        pythonize_exatts: Flag to make extra attribute names 'Pythonic'. If set
            to True, then "@stn1" will become "x_stn1".

    Returns:

    """
    attr_list = scenario.attributes("NODE")
    package = scenario.get_attribute_values("NODE", attr_list)

    node_indexer = pd.Series(package[0])
    node_indexer.index.name = 'i'
    tables = package[1:]

    if pythonize_exatts:
        attr_list = [attname.replace("@", "x_") for attname in attr_list]

    df = pd.DataFrame(index=node_indexer.index)
    for attr_name, table in zip(attr_list, tables):
        data_array = np.array(table)
        reindexed = data_array.take(node_indexer.values)
        df[attr_name] = reindexed

    df['is_centroid'] = df.index.isin(scenario.zone_numbers)

    return df


def load_link_dataframe(scenario, pythonize_exatts=False):
    """
    Creates a table for link attributes in a scenario.

    Args:
        scenario: An instance of inro.emme.scenario.Scenario
        pythonize_exatts: Flag to make extra attribute names 'Pythonic'. If set
            to True, then "@stn1" will become "x_stn1".

    Returns: pandas.DataFrame

    """
    attr_list = scenario.attributes('LINK')
    if "vertices" in attr_list: attr_list.remove("vertices")

    data_pack = scenario.get_attribute_values('LINK', attr_list)
    data_positions = data_pack[0]
    tables = data_pack[1:]

    link_indexer = {}
    for i, outgoing_data in data_positions.iteritems():
        for j, pos in outgoing_data.iteritems():
            link_indexer[(i,j)] = pos
    link_indexer = pd.Series(link_indexer)
    link_indexer.index.names = 'i j'.split()

    if pythonize_exatts:
        attr_list = [attname.replace("@", "x_") for attname in attr_list]

    df = pd.DataFrame(index= link_indexer.index)
    for attr_name, table in zip(attr_list, tables):
        data_array = np.array(table)
        reindexed = data_array.take(link_indexer.values)
        df[attr_name] = reindexed

    return df


def load_turn_dataframe(scenario, pythonize_exatts=False):
    """
    Creates a table for turn attributes in a scenario.

    Args:
        scenario: An instance of inro.emme.scenario.Scenario
        pythonize_exatts: Flag to make extra attribute names 'Pythonic'. If set
            to True, then "@stn1" will become "x_stn1".

    Returns:

    """
    attr_list = scenario.attributes("TURN")
    package = scenario.get_attribute_values("TURN", attr_list)

    index_data = package[0]
    tables = package[1:]

    turn_indexer = {}
    for (i, j), outgoing_data in index_data.iteritems():
        for k, pos in outgoing_data.iteritems():
            turn_indexer[(i,j,k)] = pos
    turn_indexer = pd.Series(turn_indexer)
    turn_indexer.index.names = "i j k".split()

    if pythonize_exatts:
        attr_list = [attname.replace("@", "x_") for attname in attr_list]

    df = pd.DataFrame(index= turn_indexer.index)
    for attr_name, table in zip(attr_list, tables):
        data_array = np.array(table)
        reindexed = data_array.take(turn_indexer.values)
        df[attr_name] = reindexed

    return df


def load_transit_line_dataframe(scenario, pythonize_exatts=False):
    """
    Creates a table for transit line attributes in a scenario.

    Args:
        scenario: An instance of inro.emme.scenario.Scenario
        pythonize_exatts: Flag to make extra attribute names 'Pythonic'. If set
            to True, then "@stn1" will become "x_stn1".

    Returns:

    """
    attr_list = scenario.attributes("TRANSIT_LINE")
    package = scenario.get_attribute_values("TRANSIT_LINE", attr_list)

    line_indexer = pd.Series(package[0])
    line_indexer.index.name = 'line'
    tables = package[1:]

    if pythonize_exatts:
        attr_list = [attname.replace("@", "x_") for attname in attr_list]

    df = pd.DataFrame(index=line_indexer.index)
    for attr_name, table in zip(attr_list, tables):
        data_array = np.array(table)
        reindexed = data_array.take(line_indexer.values)
        df[attr_name] = reindexed

    return df


def load_transit_segment_dataframe(scenario, pythonize_exatts=False):
    """
    Creates a table for transit segment attributes in a scenario.

    Args:
        scenario: An instance of inro.emme.scenario.Scenario
        pythonize_exatts: Flag to make extra attribute names 'Pythonic'. If set
            to True, then "@stn1" will become "x_stn1".

    Returns:

    """
    attr_list = scenario.attributes("TRANSIT_SEGMENT")
    package = scenario.get_attribute_values("TRANSIT_SEGMENT", attr_list)

    index_data = package[0]
    tables = package[1:]

    segment_indexer = {}
    for line, segment_data in index_data.iteritems():
        for tupl, pos in segment_data.iteritems():
            if len(tupl) == 3: i,j, loop = tupl
            else:
                i,j = tupl
                loop = 0

            segment_indexer[(line, i, j, loop)] = pos
    segment_indexer = pd.Series(segment_indexer)
    segment_indexer.index.names = "line i j loop".split()

    if pythonize_exatts:
        attr_list = [attname.replace("@", "x_") for attname in attr_list]

    df = pd.DataFrame(index=segment_indexer.index)
    for attr_name, table in zip(attr_list, tables):
        data_array = np.array(table)
        reindexed = data_array.take(segment_indexer.values)
        df[attr_name] = reindexed

    return df


def split_zone_in_matrix(base_matrix, old_zone, new_zones, proportions):
    """
    Takes a zone in a matrix (represented as a DataFrame) and splits it into several new zones,
    prorating affected cells by a vector of proportions (one value for each new zone). The old
    zone is removed.

    Args:
        base_matrix: The matrix to re-shape, as a DataFrame
        old_zone: Integer number of the original zone to split
        new_zones: List of integers of the new zones to add
        proportions: List of floats of proportions to split the original zone to. Must be the same
            length as `new_zones` and sum to 1.0

    Returns: Re-shaped DataFrame
    """

    assert isinstance(base_matrix, pd.DataFrame), "Base matrix must be a DataFrame"

    old_zone = int(old_zone)
    new_zones = np.array(new_zones, dtype=np.int32)
    proportions = np.array(proportions, dtype=np.float64)

    assert len(new_zones) == len(proportions), "Proportion array must be the same length as the new zone array"
    assert len(new_zones.shape) == 1, "New zones must be a vector"
    assert base_matrix.index.equals(base_matrix.columns), "DataFrame is not a matrix"
    assert np.isclose(proportions.sum(), 1.0), "Proportions must sum to 1.0 "

    n_new_zones = len(new_zones)

    intersection_index = base_matrix.index.drop(old_zone)
    new_index = intersection_index
    for z in new_zones: new_index = new_index.insert(-1, z)
    new_index = pd.Index(sorted(new_index))

    new_matrix = pd.DataFrame(0, index=new_index, columns=new_index, dtype=base_matrix.dtypes.iat[0])

    # 1. Copy over the values from the regions of the matrix not being updated
    new_matrix.loc[intersection_index, intersection_index] = base_matrix

    # 2. Prorate the row corresponding to the dropped zone
    # This section (and the next) works with the underlying Numpy arrays, since they handle
    # broadcasting better than Pandas does
    original_row = base_matrix.loc[old_zone, intersection_index]
    original_row = original_row.values[:] # Make a shallow copy to preserve shape of the original data
    original_row.shape = 1, len(intersection_index)
    proportions.shape = n_new_zones, 1
    result = pd.DataFrame(original_row * proportions, index=new_zones, columns=intersection_index)
    new_matrix.loc[result.index, result.columns] = result

    # 3. Proprate the column corresponding to the dropped zone
    original_column = base_matrix.loc[intersection_index, old_zone]
    original_column = original_column.values[:]
    original_column.shape = len(intersection_index), 1
    proportions.shape = 1, n_new_zones
    result = pd.DataFrame(original_column * proportions, index=intersection_index, columns=new_zones)
    new_matrix.loc[result.index, result.columns] = result

    # 4. Expand the old intrazonal
    proportions_copy = proportions[:,:]
    proportions_copy.shape = 1, n_new_zones
    proportions.shape = n_new_zones, 1

    intrzonal_matrix = proportions * proportions_copy
    intrazonal_scalar = base_matrix.at[old_zone, old_zone]

    result = pd.DataFrame(intrazonal_scalar * intrzonal_matrix, index=new_zones, columns=new_zones)
    new_matrix.loc[result.index, result.columns] = result

    return new_matrix


def matrix_to_pandas(mtx, scenario_id=None):
    """
    Converts Emme Matrix objects to Pandas Series or DataFrames. Origin and Destination matrices will be
    converted to Series, while Full matrices will be converted to DataFrames. Scalar matrices are unsupported.

    Args:
        mtx: Either a Matrix object or a MatrixData object
        scenario_id: Int, optional. Must be provided if a `mtx` is a Matrix object.

    Returns: Series or DataFrame, depending on the type of matrix.

    """
    if hasattr(mtx, 'prefix'): # Duck typing check for Matrix object rather than Matrix Data
        assert mtx.type != 'SCALAR', "Scalar matrices cannot be converted to DataFrames"
        md = mtx.get_data(scenario_id)
    else: md = mtx

    zones_tupl = md.indices
    if len(zones_tupl) == 1:
        # Origin or Destination matrix
        idx = pd.Index(zones_tupl[0])
        idx.name = 'zone'
        vector = md.to_numpy()
        return pd.Series(vector, index=idx)
    elif len(zones_tupl) == 2:
        # Full matrix
        idx = pd.Index(zones_tupl[0])
        idx.name = 'p'
        cols = pd.Index(zones_tupl[1])
        cols.name = 'q'
        matrix = md.to_numpy()
        return pd.DataFrame(matrix, index=idx, columns=cols)
    else:
        raise ValueError("Could not infer matrix from object type %s", repr(mtx))


def pandas_to_matrix(series_or_dataframe, mtx_out=None, scenario_id=None):
    """
    Converts a Series or DataFrame to an Emme Matrix; either as a MatrixData instance or saved to a Matrix instance.

    Args:
        series_or_dataframe: The Series or DataFrame to convert.
        mtx_out: A Matrix instance (usually retrieved from Emmebank.matrix()) or None. If provided, the resulting matrix
            data will be saved to the Emmebank
        scenario_id: Optional scenario ID to pass to Matrix.set_data()

    Returns:
        MatrixData if `mtx_out` is None; None otherwise.

    """
    if isinstance(series_or_dataframe, pd.Series):
        indices = list(series_or_dataframe.index.values)
        md = MatrixData(indices)
        md.from_numpy(series_or_dataframe.values)
    elif isinstance(series_or_dataframe, pd.DataFrame):
        indices = list(series_or_dataframe.index.values), list(series_or_dataframe.columns.values)
        md = MatrixData(indices)
        md.from_numpy(series_or_dataframe.values)
    else: raise TypeError("Expected a Series or DataFrame, got %s" %type(series_or_dataframe))

    if mtx_out is not None:
        mtx_out.set_data(md, scenario_id=scenario_id)
        return
    return md


def parallel_strategy_allowed(logr):
    # Check for Emme 4.3 OR for a test-beta version. By default, if there's a problem, turn off parallel analysis
    try:
        if os.environ['EMMEPATH'].endswith('Emme-test-160811'):
            logr.warn("Detected test version of Emme base on EMMEPATH environmental variable. Turning on parallel analysis, "
                      "but be aware that this check may not be future-proof")
            return True
        elif mm.desktop.version_info >= (4,3):
            logr.warn("Emme 4.3 detected. Turning on parallel analysis")
            return True
    except Exception as e:
        # If there's some error in checking version, leave PARALLEL_ANALYSIS turned off by default
        logr.error("Error checking Emme version: %s" % e)
        return False