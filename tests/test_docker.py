from __future__ import annotations

import pytest

docker = pytest.importorskip("docker")


@pytest.mark.requires_docker
def test_docker_daemon_accessible(docker_client) -> None:
    """Ensure we can talk to the Docker daemon.

    If the daemon is not running or unavailable the test is skipped by
    ``docker_client`` in ``conftest`` which also emits a warning.
    """

    assert docker_client.ping() is True
