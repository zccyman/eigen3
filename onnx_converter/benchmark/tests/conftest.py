# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/03/29 19:41:02
# @File    : conftest.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

from datetime import datetime

import pytest
from py._xmlgen import html


def pytest_addoption(parser):
    parser.addoption(
        "--model_dir", default="./trained_models", help="model root directory"
    )
    parser.addoption(
        "--dataset_dir", default='/buffer', help="dataset root directory"
    )
    parser.addoption(
        "--selected_mode", default='MR_RELEASE', help="test all model in each task, when selected_mode equal 'MR_RELEASE' "
    )
    parser.addoption(
        "--password", default=None, help="set password"
    )

@pytest.fixture
def model_dir(request):
    return request.config.getoption("--model_dir")


@pytest.fixture
def dataset_dir(request):
    return request.config.getoption("--dataset_dir")


@pytest.fixture
def password(request):
    return request.config.getoption("--password")


@pytest.fixture
def selected_mode(request):
    return request.config.getoption("--selected_mode")


@pytest.mark.optionalhook
def pytest_html_results_table_header(cells):
    cells.insert(2, html.th('Time', class_='sortable time', col='time'))
    cells.pop()


@pytest.mark.optionalhook
def pytest_html_results_table_row(report, cells):
    cells.insert(2, html.td(datetime.utcnow(), class_='col-time'))
    cells.pop()
    if report.skipped:
        del cells[:]


def pytest_configure(config):
    # add Project/Converter
    config._metadata["Project"] = "Converter Test Benchmark"
    config._metadata['Converter'] = 'https://192.168.1.240/vision-algorithm/onnx-converter'
    # delete Python
    # config._metadata.pop("Python")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(report, "duration_formatter", "%H:%M:%S.%f")
