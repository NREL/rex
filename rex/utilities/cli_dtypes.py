# -*- coding: utf-8 -*-
"""
Custom dtypes for Click.
"""
import click
import logging


logger = logging.getLogger(__name__)


def sanitize_str(value, subs=('slice', '=', '(', ')', ' ')):
    """Sanitize characters from string."""
    for s in subs:
        value = value.replace(s, '')
    return value


class IntType(click.ParamType):
    """Integer click input argument type with option for None."""
    name = 'int'

    def convert(self, value, param, ctx):
        """Convert to int or return as None."""
        if isinstance(value, str):
            if 'None' in value:
                return None
            else:
                return int(value)
        elif isinstance(value, int):
            return value
        else:
            self.fail('Cannot recognize int type: {} {}'
                      .format(value, type(value)), param, ctx)


class FloatType(click.ParamType):
    """Float click input argument type with option for None."""
    name = 'float'

    def convert(self, value, param, ctx):
        """Convert to float or return as None."""
        if isinstance(value, str):
            if 'None' in value:
                return None
            else:
                return float(value)
        elif isinstance(value, float):
            return value
        else:
            self.fail('Cannot recognize float type: {} {}'
                      .format(value, type(value)), param, ctx)


class StrType(click.ParamType):
    """String click input argument type with option for None."""
    name = 'str'

    def convert(self, value, param, ctx):
        """Convert to str or return as None."""
        if isinstance(value, str):
            if value.lower() == 'none':
                return None
            else:
                return value
        else:
            self.fail('Cannot recognize str type: {} {}'
                      .format(value, type(value)), param, ctx)


class StrFloatType(click.ParamType):
    """String or float click input argument type with option for None."""
    name = 'strfloat'

    def convert(self, value, param, ctx):
        """Convert to str or float or return as None."""
        if isinstance(value, str):
            if value.lower() == 'none':
                return None
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
                return value
        else:
            self.fail('Cannot recognize str or float type: {} {}'
                      .format(value, type(value)), param, ctx)


class ListType(click.ParamType):
    """Base list click input argument type with option for None."""
    name = 'list'

    @staticmethod
    def dtype(x):
        """Option to enforce a Homogeneous datatype."""
        return x

    def convert(self, value, param, ctx):
        """Convert string to list."""
        if isinstance(value, str):
            value = sanitize_str(value, subs=['=', '(', ')', ' ', '[', ']',
                                              '"', "'"])
            if value.lower() == 'none':
                return None
            list0 = value.split(',')
            return [self.dtype(x) for x in list0]
        elif isinstance(value, list):
            return value
        elif isinstance(value, type(None)):
            return value
        else:
            self.fail('Cannot recognize list type: {} {}'
                      .format(value, type(value)), param, ctx)


class FloatListType(ListType):
    """Homogeneous list of floats click input argument type."""
    name = 'floatlist'

    @staticmethod
    def dtype(x):
        """Enforce a homogeneous float datatype."""
        return float(x)


class IntListType(ListType):
    """Homogeneous list of integers click input argument type."""
    name = 'intlist'

    @staticmethod
    def dtype(x):
        """Enforce a homogeneous integer datatype."""
        return int(x)


class StrListType(ListType):
    """Homogeneous list of strings click input argument type."""
    name = 'strlist'

    @staticmethod
    def dtype(x):
        """Enforce a homogeneous string datatype."""
        return str(x)


class PathListType(ListType):
    """Homogeneous list of click path input argument types."""
    name = 'pathlist'

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            kwargs for click.types.Path
        """
        self._click_path_type = click.types.Path(**kwargs)

    def convert(self, value, param, ctx):
        """Convert string to list."""
        if isinstance(value, str):
            value = sanitize_str(value, subs=['=', '(', ')', ' ', '[', ']',
                                              '"', "'"])
            if value.lower() == 'none':
                return None
            list0 = value.split(',')
            return [self.path(x, param, ctx) for x in list0]
        elif isinstance(value, list):
            return value
        elif isinstance(value, type(None)):
            return value
        else:
            self.fail('Cannot recognize list type: {} {}'
                      .format(value, type(value)), param, ctx)

    def path(self, value, param, ctx):
        """Enforce a homogeneous string datatype."""
        return self._click_path_type.convert(value, param, ctx)


class StrOrListType(click.ParamType):
    """Flexible type for string or list of strings."""
    name = 'str_or_list'

    @staticmethod
    def dtype(x):
        """Enforce a homogeneous string datatype."""
        return str(x)

    def convert(self, value, param, ctx):
        """Convert string to list."""
        if isinstance(value, str):
            if (('(' in value and ')' in value)
                    or ('[' in value and ']' in value)):
                value = sanitize_str(value, subs=['=', '(', ')', ' ', '[', ']',
                                                  '"', "'"])
                list0 = value.split(',')
                return [self.dtype(x) for x in list0]
            else:
                if value.lower() == 'none':
                    return None
                else:
                    return value
        elif isinstance(value, list):
            return value
        elif isinstance(value, type(None)):
            return value
        else:
            self.fail('Cannot recognize list type: {} {}'
                      .format(value, type(value)), param, ctx)


INT = IntType()
FLOAT = FloatType()
STR = StrType()
STRFLOAT = StrFloatType()
INTLIST = IntListType()
FLOATLIST = FloatListType()
STRLIST = StrListType()
PATHLIST = PathListType()
STR_OR_LIST = StrOrListType()
