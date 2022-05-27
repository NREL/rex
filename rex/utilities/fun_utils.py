# -*- coding: utf-8 -*-
"""
Utilities for parsing function signatures to make easy CLI's or to call
functions programmatically
"""
import inspect
from inspect import signature
import logging

logger = logging.getLogger(__name__)


def arg_to_str(arg):
    """Format input as str w/ appropriate quote types for python call

    Returns
    --------
    out : str
        String rep of the input arg with proper quotation for formatting in
        python -c '{commands}' cli calls. For example, int, float, None -> '0'
        str -> \"string\"
    """
    if isinstance(arg, str):
        return f'"{arg}"'
    else:
        return f'{arg}'


def has_class(obj):
    """Determine whether an object is a method that is bound to a class

    Returns
    -------
    out : bool
        Whether the input object belongs to a class. For example,
        MyClass.bound_method will return True, whereas standalone_fun will
        return False.
    """
    if hasattr(obj, '__qualname__'):
        if len(obj.__qualname__.split('.')) > 1:
            return True
    else:
        return False


def get_class(obj):
    """Get the name of the class that the method object is bound to. Returns an
    empty string if the method object is not bound to a class.

    Returns
    -------
    out : str
        The class name the input object method belongs to. For example,
        MyClass.bound_method will return "MyClass", whereas standalone_fun will
        return an empty string.
    """
    class_name = ''
    if has_class(obj):
        class_name = obj.__qualname__.split('.')[0]

    return class_name


def is_standalone_fun(obj):
    """Determine whether an object is a standalone function without a class

    Returns
    -------
    out : bool
        Whether the input object is a standalone function. For example,
        MyClass.bound_method will return False, whereas standalone_fun will
        return True.
    """
    return inspect.isfunction(obj) and not has_class(obj)


def get_fun_str(fun):
    """Get the function string from a function object including the
    ClassName.function if the function is bound

    Returns
    -------
    out : str
        The function string to call the input function. For example
        MyClass.bound_method will return "MyClass.bound_method", whereas
        standalone_fun will return "standalone_fun".
    """
    fun_name = fun.__name__
    if is_standalone_fun(fun):
        return fun_name
    elif has_class(fun):
        class_name = get_class(fun)
        return f'{class_name}.{fun_name}'
    else:
        return fun_name


def get_arg_str(fun, config):
    """Get a string representation of positional and keyword arguments required
    by an input function and provided in the config dictionary.

    Example
    -------
    If the function signature is my_fun(a, b, c=0) and config is
    {'a': 1, 'b': 2, 'c': 3}, the returned arg_str will be "1, 2, c=3". The
    function can also take *args or **kwargs, which will be taken from the
    "args" and "kwargs" keys in the config. "args" must be mapped to a list,
    and "kwargs" must be mapped to a dictionary.

    Parameters
    ----------
    fun : obj
        A callable object with a function signature. The function signature
        will be parsed for args and kwargs which will be taken from the config.
    config : dict
        A namespace of arguments to run fun. Not all entries in config may be
        used, but all required inputs to fun must be provided in config. Can
        include "args" and "kwargs" which must be mapped to a list and a
        dictionary, respectively.

    Returns
    -------
    arg_str : str
        Argument string that can be used to call fun programmatically, e.g.
        fun(arg_str)
    """

    sig = signature(fun)

    arg_strs = []

    for arg_name, value in sig.parameters.items():
        is_self = arg_name == 'self'
        is_kw = value.default != value.empty
        is_star_arg = str(value).startswith('*') and str(value).count('*') == 1
        is_star_kwa = str(value).startswith('*') and str(value).count('*') == 2
        not_required = (is_self or is_kw or is_star_arg or is_star_kwa)
        required = not not_required

        if arg_name in config:
            if not is_kw and not (is_star_arg or is_star_kwa):
                arg = arg_to_str(config[arg_name])
                arg_strs.append(f'{arg}')

            elif is_kw and not (is_star_arg or is_star_kwa):
                arg = arg_to_str(config[arg_name])
                arg_strs.append(f'{arg_name}={arg}')

            elif is_star_arg:
                msg = '"args" key in config must be mapped to a list!'
                assert isinstance(config[arg_name], (list, tuple)), msg
                arg_strs += [f'{arg_to_str(star_arg)}'
                             for star_arg in config[arg_name]]

            elif is_star_kwa:
                msg = '"kwargs" key in config must be mapped to a dict!'
                assert isinstance(config[arg_name], dict), msg
                arg_strs += [f'{star_name}={arg_to_str(star_kw)}'
                             for star_name, star_kw
                             in config[arg_name].items()]

        elif required:
            msg = (f'Positional argument "{arg_name}" '
                   'needs to be defined in config!')
            logger.error(msg)
            raise KeyError(msg)

    arg_str = ', '.join(arg_strs)

    return arg_str


def get_fun_call_str(fun, config, quote_char='\"'):
    """Get a string that will call a function using args and kwargs from a
    generic config.

    Example
    -------
    If the function signature is my_fun(a, b, c=0) and config is
    {'a': 1, 'b': 2, 'c': 3}, the returned call string will be
    "my_fun(1, 2, c=3)". The function can also take *args or **kwargs, which
    will be taken from the "args" and "kwargs" keys in the config. "args" must
    be mapped to a list, and "kwargs" must be mapped to a dictionary.

    Parameters
    ----------
    fun : obj
        A callable object with a function signature. The function signature
        will be parsed for args and kwargs which will be taken from the config.
    config : dict
        A namespace of arguments to run fun. Not all entries in config may be
        used, but all required inputs to fun must be provided in config. Can
        include "args" and "kwargs" which must be mapped to a list and a
        dictionary, respectively.
    quote_char : str
        Character to use for string quotes in the fun_call_str output.

    Returns
    -------
    fun_call_str : str
        A string representation of a function call e.g. "fun(arg1, arg2,
        kw1=kw1)" where arg1, arg2, and kw1 were found in the config.
    """

    fun_str = get_fun_str(fun)
    arg_str = get_arg_str(fun, config)
    call_str = f'{fun_str}({arg_str})'

    if quote_char:
        call_str = call_str.replace('"', quote_char)
        call_str = call_str.replace("'", quote_char)
        call_str = call_str.replace("/'", quote_char)
        call_str = call_str.replace('/"', quote_char)

    return call_str
