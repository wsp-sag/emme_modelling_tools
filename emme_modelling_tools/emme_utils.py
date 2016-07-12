from __future__ import division

from contextlib import contextmanager

from shapely.geometry import Point, LineString, Polygon
import shapelib as shp
import dbflib as dbf
from os import path

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
