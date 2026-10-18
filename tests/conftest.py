import pytest

from ragsec.corpus import build_world


@pytest.fixture()
def world():
    return build_world(3, poison=True)


@pytest.fixture()
def people(world):
    """(tenant, user) -> Principal"""
    return {(p.tenant, p.user): p for p in world.principals}
