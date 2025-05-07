import pytest
from convoke.parsers import parse_numbered_list, parse_json_list


def test_parse_numbered_list_empty():
    assert parse_numbered_list("") == []
    assert parse_numbered_list(None) == []


def test_parse_numbered_list_non_string():
    assert parse_numbered_list(123) == []


def test_parse_numbered_list_malformed():
    text = "1. Foo\nBar\n* Baz: Desc\n  More"
    items = parse_numbered_list(text)
    assert items[0][0] == "Foo"
    assert "Bar" in items[0][1]
    assert items[1][0] == "Baz"


def test_parse_json_list_empty():
    assert parse_json_list("") == []
    assert parse_json_list(None) == []


def test_parse_json_list_non_string():
    assert parse_json_list(123) == []


def test_parse_json_list_malformed():
    assert parse_json_list("not a json") == []
    assert parse_json_list("[{'name': 'A'}]") == []


def test_parse_json_list_partial():
    text = '[{"name": "A", "description": "desc"}, {"foo": 1}]'
    items = parse_json_list(text)
    assert items == [("A", "desc")]
