from __future__ import division

from collections import OrderedDict
import json
from keyword import kwlist
import os
import re
from StringIO import StringIO

special_chars = set(r" .,<>/?;:'|[{]}=+-)(*&^%$#@!`~" + '"')
regex_chars = set(r"]")
pyspecchar = list(special_chars - regex_chars)
escaped_chars = ["\%s" % c for c in regex_chars]
insertion = ''.join(pyspecchar + escaped_chars)
UNPYTHONIC_REGEX = re.compile(r"^\d|[%s\s]+" % insertion)
del pyspecchar, escaped_chars, insertion, special_chars, regex_chars


def is_name_pythonic(name):
    return not UNPYTHONIC_REGEX.match(name) or name in kwlist


def load_commented_json(reader, **kwargs):
    if isinstance(reader, basestring):
        with open(reader) as file_:
            parsed = _parse_comments(file_)
    else:
        parsed = _parse_comments(reader)

    return json.loads(parsed, **kwargs)


def _parse_comments(reader):
    regex = r'\s*(#|\/{2}).*$'
    regex_inline = r'(:?(?:\s)*([A-Za-z\d\.{}]*)|((?<=\").*\"),?)(?:\s)*(((#|(\/{2})).*)|)$'

    pipe = []
    for line in reader:
        if re.search(regex, line):
            if re.search(r'^' + regex, line, re.IGNORECASE):
                continue
            elif re.search(regex_inline, line):
                pipe.append(re.sub(regex_inline, r'\1', line))
        else:
            pipe.append(line)
    return "\n".join(pipe)


class ConfigParseError(IOError):
    pass


class ConfigSpecificationError(AttributeError):
    pass


class Config(object):
    """A class to manage model configuration files."""

    @staticmethod
    def fromfile(fp):
        dict_ = load_commented_json(fp, object_pairs_hook=OrderedDict)
        root_name, _ = os.path.splitext(os.path.basename(fp))
        return Config(dict_, file=fp, name=root_name)

    @staticmethod
    def fromstring(s, file_name='<from_str>', root_name='<root>'):
        sio = StringIO(s)
        dict_ = load_commented_json(sio, object_pairs_hook=OrderedDict)
        return Config(dict_, file=file_name, name=root_name)

    @staticmethod
    def fromdict(dict_, file_name='<from_dict>', root_name='<root>'):
        return Config(dict_, file=file_name, name=root_name)

    def __init__(self, config_dict, name=None, parent=None, file=None):
        self._d = OrderedDict()
        self._attrs = set()
        self._name = name
        self._parent = parent
        self._file = file

        existing_vals = set(dir(self))
        for key, val in config_dict.iteritems():
            if key in existing_vals:
                raise ConfigParseError(key)

            if isinstance(val, dict):
                val = Config(val, key, self, file)
            elif isinstance(val, list):
                val = [
                    Config(item, key + "[%s]" % i, self, file)
                    if isinstance(item, dict)
                    else item
                    for i, item in enumerate(val)
                ]

            if not is_name_pythonic(key):
                self._d[key] = val
            else:
                setattr(self, key, val)
                self._attrs.add(key)
                self._d[key] = val  # Also copy to the dict
        self._attrs = frozenset(self._attrs)

    @property
    def asdict(self):
        """Getter for the dict of un-pythonic names"""
        return self._d

    @property
    def attrs(self):
        return self._attrs

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def namespace(self):
        name = self._name if self._name is not None else '<unnamed>'
        if self._parent is None:
            return name
        return '.'.join([self._parent.namespace, name])

    def __str__(self):
        if self._parent is None:
            return "Config @%s" % self._file

        return "Config(%s) @%s" % (self.namespace, self._file)

    def __getattr__(self, item):
        raise ConfigSpecificationError("Item '%s' is missing from config <%s>" % (item, self.namespace))

    def __contains__(self, item):
        return item in self._d

    def serialize(self):
        child_dict = OrderedDict()
        for attr, item in self._d.iteritems():
            if isinstance(item, Config):
                child_dict[attr] = item.serialize()
            elif isinstance(item, list):
                child_dict[attr] = [x.serialize() if isinstance(x, Config) else x for x in item]
            else:
                child_dict[attr] = item
        return child_dict

    def tofile(self, fp):
        dict_ = self.serialize()
        with open(fp, 'w') as writer:
            json.dump(dict_, writer, sort_keys=True, indent=2)
