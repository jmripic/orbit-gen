# import pytest
# import spiceypy as spice

# @pytest.fixture(scope="session")
# def em_mu_star():
#     spice.furnsh("")


#     gm_primary = spice.bodvrd("EARTH", "GM", 1)[1][0]
#     gm_secondary = spice.bodvrd("MOON", "GM", 1)[1][0]
#     mu_star = gm_secondary / (gm_primary + gm_secondary)
