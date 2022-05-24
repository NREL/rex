# -*- coding: utf-8 -*-
"""
Utilities for building efficient command line interfaces on Click
"""
import pytest
from rex.utilities.fun_utils import (has_class, get_class, is_standalone_fun,
                                     get_fun_str, get_fun_call_str)


# pylint: disable=unused-argument
def my_fun(a, b, *args, c=0, d=1, **kwargs):
    """Test standalone function"""
    out = a + b + c + d
    for v in kwargs.values():
        out += v
    return out


# pylint: disable=unused-argument
def my_str_fun(a, b, c=None, d='test'):
    """Test string and None args"""
    return None


class MyClass:
    """Test class"""
    A = 0.0001

    def __init__(self, a):
        self.a = a

    def inst_meth(self, b, c=0, d=1):
        """Test instance method"""
        return self.a + b + c + d

    @staticmethod
    def static_meth(b, c=0, d=1):
        """Test static method"""
        return b + c + d

    @classmethod
    def cls_meth(cls, b, c=0, d=1):
        """Test class method"""
        return cls.A + b + c + d


def test_has_class():
    """Test the boolean has_class() method"""
    myclass = MyClass(0)
    assert not has_class(my_fun)
    assert has_class(MyClass.inst_meth)
    assert has_class(MyClass.static_meth)
    assert has_class(MyClass.cls_meth)
    assert has_class(myclass.inst_meth)
    assert has_class(myclass.static_meth)
    assert has_class(myclass.cls_meth)


def test_get_class():
    """Test the class string retrieval method"""
    myclass = MyClass(0)
    assert get_class(my_fun) == ''
    assert get_class(MyClass.inst_meth) == 'MyClass'
    assert get_class(MyClass.static_meth) == 'MyClass'
    assert get_class(MyClass.cls_meth) == 'MyClass'
    assert get_class(myclass.inst_meth) == 'MyClass'
    assert get_class(myclass.static_meth) == 'MyClass'
    assert get_class(myclass.cls_meth) == 'MyClass'


def test_is_standalone():
    """Test the method to tell if an obj is a standalone function"""
    myclass = MyClass(0)
    assert is_standalone_fun(my_fun)
    assert not is_standalone_fun(MyClass.inst_meth)
    assert not is_standalone_fun(MyClass.static_meth)
    assert not is_standalone_fun(MyClass.cls_meth)
    assert not is_standalone_fun(myclass.inst_meth)
    assert not is_standalone_fun(myclass.static_meth)
    assert not is_standalone_fun(myclass.cls_meth)


def test_get_fun_str():
    """Test the function string retrieval method"""
    myclass = MyClass(0)
    assert get_fun_str(my_fun) == 'my_fun'
    assert get_fun_str(MyClass.inst_meth) == 'MyClass.inst_meth'
    assert get_fun_str(MyClass.static_meth) == 'MyClass.static_meth'
    assert get_fun_str(MyClass.cls_meth) == 'MyClass.cls_meth'
    assert get_fun_str(myclass.inst_meth) == 'MyClass.inst_meth'
    assert get_fun_str(myclass.static_meth) == 'MyClass.static_meth'
    assert get_fun_str(myclass.cls_meth) == 'MyClass.cls_meth'


def test_bad_args():
    """Test bad function and config arguments"""
    config = {}

    # missing required argument b
    with pytest.raises(KeyError):
        _ = get_fun_call_str(MyClass.inst_meth, config)

    # missing required argument b
    with pytest.raises(KeyError):
        _ = get_fun_call_str(MyClass.static_meth, config)


def test_get_call_str():
    """Test the retrieval of a full function call string"""
    config = {'a': 0, 'b': 2.2, 'c': 3.4,
              'args': [4.01, 1.01],
              'kwargs': {'d': 1}}

    call_str = get_fun_call_str(my_fun, config)
    assert call_str == 'my_fun(0, 2.2, 4.01, 1.01, c=3.4, d=1)'

    call_str = get_fun_call_str(MyClass.cls_meth, config)
    assert call_str == 'MyClass.cls_meth(2.2, c=3.4)'

    call_str = get_fun_call_str(MyClass.static_meth, config)
    assert call_str == 'MyClass.static_meth(2.2, c=3.4)'

    config = {'a': 0, 'b': 2.2}
    call_str = get_fun_call_str(MyClass.static_meth, config)
    assert call_str == 'MyClass.static_meth(2.2)'


def test_init_str():
    """Test the parsing of a class init function"""
    assert get_fun_str(MyClass) == 'MyClass'
    config = {'a': 0, 'b': 2.2}
    call_str = get_fun_call_str(MyClass, config)
    assert call_str == 'MyClass(0)'


def test_str_args():
    """Test parsing of non-numeric string and None arguments"""
    config = {'a': {'test': 'x'}, 'b': 'y', 'd': None}
    call_str = get_fun_call_str(my_str_fun, config)
    assert call_str == 'my_str_fun({\"test\": \"x\"}, \"y\", d=None)'
