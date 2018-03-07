#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true",
                     help="skip slow tests")
    parser.addoption("--only-slow", action="store_true",
                     help="run only slow tests")
    parser.addoption("--skip-examples", action="store_true",
                     help="skip tests of examples")

    parser.addoption("--skip-smoke", action="store_true",
                     help="skip tests marked as smoke (i.e. smoke-tests)")
    parser.addoption("--skip-not_vetted", action="store_true",
                     help="skip tests marked as not_vetted")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and item.config.getoption("--skip-slow"):
        pytest.skip("skipping due to --skip-slow")

    if 'slow' not in item.keywords and item.config.getoption("--only-slow"):
        pytest.skip("skipping due to --only-slow")

    if 'example' in item.keywords and item.config.getoption("--skip-examples"):
        pytest.skip("skipping due to --skip-examples")

    if 'smoke' in item.keywords and item.config.getoption("--skip-smoke"):
        pytest.skip("skipping due to --skip-smoke")

    if ('not_vetted' in item.keywords and
            item.config.getoption("--skip-not_vetted")):
        pytest.skip("skipping due to --skip-not_vetted")
