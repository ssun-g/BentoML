from __future__ import annotations

import os
import typing as t
import tempfile
import contextlib
from typing import TYPE_CHECKING

import numpy as np
import psutil
import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest as _PytestFixtureRequest
    from _pytest.config import Config

    class FixtureRequest(_PytestFixtureRequest):
        param: str


@pytest.fixture()
def img_file(tmpdir) -> str:
    import PIL.Image

    img_file_ = tmpdir.join("test_img.bmp")
    img = PIL.Image.fromarray(np.random.randint(255, size=(10, 10, 3)).astype("uint8"))
    img.save(str(img_file_))
    return str(img_file_)


@pytest.fixture()
def bin_file(tmpdir: str) -> str:
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)


def pytest_configure(config: Config) -> None:  # pylint: disable=unused-argument
    import sys
    import subprocess

    cmd = f"{sys.executable} {os.path.join(os.getcwd(), 'train.py')}"
    subprocess.run(cmd, shell=True, check=True)

    # use the local bentoml package in development
    os.environ["BENTOML_BUNDLE_LOCAL_BUILD"] = "True"
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"


def pytest_sessionstart(session: Session):  # pylint: disable=unused-argument
    from bentoml._internal.models import ModelStore

    path = tempfile.mkdtemp("bentoml-pytest-e2e")
    from bentoml._internal.configuration.containers import BentoMLContainer

    BentoMLContainer.model_store.set(ModelStore(path))


@pytest.fixture(scope="session", autouse=True)
def clean_context() -> t.Generator[contextlib.ExitStack, None, None]:
    stack = contextlib.ExitStack()
    yield stack
    stack.close()


@pytest.fixture(
    params=[
        "server_config_default.yml",
        "server_config_cors_enabled.yml",
    ],
    scope="session",
)
def server_config_file(request):
    return request.param


@pytest.fixture(
    params=[
        "standalone",
        "docker",
        "distributed",
    ],
    scope="session",
)
def deployment_mode(request: FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def host(
    deployment_mode: str,
    server_config_file: str,
    clean_context: contextlib.ExitStack,
) -> t.Generator[str, None, None]:
    if (
        (psutil.WINDOWS or psutil.MACOS)
        and os.environ.get("GITHUB_ACTION")
        and deployment_mode == "docker"
    ):
        pytest.skip(
            "due to GitHub Action's limitation, docker deployment is not supported on "
            "windows/macos. But you can still run this test on macos/windows locally."
        )

    if not psutil.LINUX and deployment_mode == "distributed":
        pytest.skip("distributed deployment is only supported on Linux")

    from bentoml.testing.server import host_bento

    with host_bento(
        "service:svc",
        config_file=server_config_file,
        deployment_mode=deployment_mode,
        clean_context=clean_context,
    ) as host:
        yield host
