import os.path, pathlib, pickle, scipy.io
import tools.constants as SysData
import tools.plot_tools as pt


class Solution:
    """This class is meant to provide functionality to manipulate a KNOWN and provided state solution for an orbit. """
    def __init__(self, path_str):
        """Initialize a solution class instance by loading a pickle .p or a matlab .mat file of the orbit's timeseries of the state vector.
        
        Args:
            path_str (string): String containing the path to the data file the user wants to load
            
        Returns:
            None. """

        match pathlib.Path(path_str).suffix:
            # We would like to use both pickle .p and matlab .mat data files, so this block provides more robust handling of file types. Functionality assumes that the loaded data of the pickle and mat files have the same key-value pairings.
            case '.p':
                # Read the pickle file and assign contents to variable 'data'. Having it in self.data allows for continuous access to the raw data after instantiation.
                path_f1 = os.path.normpath(os.path.expandvars(path_str))
                with open(path_f1, 'rb') as file:
                    self.data = pickle.load(file)   # 'data' is an attribute of the Solution class

            case '.mat':
                # Read the matlab file and assign contents to variable 'data'. Having it in self.data allows for continuous access to the raw data after instantiation.
                path_f1 = os.path.normpath(os.path.expandvars(path_str))
                with open(path_f1, 'rb') as file:
                    self.data = scipy.io.loadmat(file)

        self.construct_vars()       # Assign data to relevant variables. See below.
        self.dimensioned = True     # Track if the data has units/dimensions. True if they do. 
        pass


    def construct_vars(self):
        """This is a construct function that assigns the relevant data from the loaded file to class variables. Simply meant to move these lines of code out of the __init__() function."""

        self.statevec = self.data['state']   # statevec: position [km] and velocity [km/s]; nx6 vector
        self.tvec = self.data['t']           # tvec: times where states are defined, [seconds]
        self.period = self.data['t'][-1]     # period: orbit period [s]; assumes solution is ONLY one orbit
        self.state0 = self.statevec[0,:]     # initial state vector; 1x6

        print(f'Initial state x0: {self.state0} with Period: {self.period}')
        return

    
    def nondimensionalize(self, args={'redimensionalize':False}):
        """Non-dimensionalize the data from the pickle file to use standard CR3BP practices. Can be useful but not always desired, so it is available as a method of the Solution class parameter.
        
        Args:
            redimensionalize (boolean): Remove or replace the dimensions of the data. If False, then nondimensionalize the system to the unitless canonical CR3BP formulation. If True, then re-apply dimensions to the system."""

        _args = {'redimensionalize':False} # Default arguments. Maintains original args variable
        for key in args.keys():
            _args[ key ] = args[ key ]

        if _args['redimensionalize'] == False : # Execute if no argument is passed, or if arg is passed as default
            self.statevec[:,0:3] = self.statevec[:,0:3] /SysData.lstar
            self.statevec[:,3:] = self.statevec[:,3:] *SysData.tstar/SysData.lstar
            self.tvec = self.tvec /SysData.tstar
            self.period = self.period /SysData.tstar
            self.dimensioned = False
            return
        
        else : # Redimensionalize the system
            self.statevec[:,0:3] = self.statevec[:,0:3] *SysData.lstar
            self.statevec[:,3:] = self.statevec[:,3:] /SysData.tstar*SysData.lstar
            self.tvec = self.tvec *SysData.tstar
            self.period = self.period *SysData.tstar
            self.dimensioned = True
            return
        
    
    def plot_orbit(self,args={}):
        """Plot the loaded orbit in 3D.
        args={'Frame': 'Synodic', 'dimensioned':True}. Args used to specify plotting specs."""

        _args = {'Frame': 'Synodic', 'dimensioned':self.dimensioned}
        for key in args.keys():
            _args[ key ] = args[ key ]
            
        pt.Orbit3D(self.statevec[:,0:3], self.tvec, _args)
        return
