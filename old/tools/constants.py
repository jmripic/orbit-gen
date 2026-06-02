"""Inventory of numerical constants, their units, and what they are used for."""

G = 6.6743*10**(-20)            # Grav Constant

# Earth Constants
earthMass = 5.97217 * 10 ** 24  # Earth mass (kg)
earthD = 12742                  # Earth diameter 
earthR = 6371                   # Earth radius (km)
AU = 1.495978707 * 10 ** 8      # Astronaomical Unit (km)

# Moon Constants
moonMass = .07346 * 10 ** 24    # Moon mass 
moonD = 3474.8                  # Moon diameter (km)
moonR = 1737.400                # Moon radius (km), because I am quite lazy
moonSMA = 384400                # Moon semi major axis around Earth [km]. Used for dimensionalizing
# Earth-Moon System Values
mustar = moonMass/(moonMass + earthMass) # specific mu of the three-body system, 1.215058560962404E-2
lstar = 389703 # Length unit of Earth-Moon system (km). Used for the nondimensionalizing EOM's
tstar = 382981 # Time unit of Earth-Moon system (s). Used for the nondimensionalizing EOM's
EMmu = G*(earthMass+moonMass)


# Sun Constants
ws = 0.925195985520347 # taken from KoonLoMarRoss 2011
sunMass = (1988500 * 10 ** 24)/ (earthMass + moonMass) # Non-dimensionalized mass of of sun in EM system: ms/(me+mm)
