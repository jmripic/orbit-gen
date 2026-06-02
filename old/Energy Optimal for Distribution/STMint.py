from sympy import *
import numpy as np
from scipy.integrate import solve_ivp
from astropy import constants as const
from astropy import units as u


class STMint:
    """State Transition Matrix Integrator
    A tool for numerical integration of variational
    equations associated with a symbolically specified dynamical system.

    Constructor Parameters:
        vars (1-dimensional sympy matrix):
            The variables used in the symbolic integration.
        dynamics (sympy expression(s)):
            The dynamics to be symbolically integrated.
        preset (string):
            Dynamic and Variational equation preset. Current presets are:
                "twoBody":
                    Two body motion.
                "twoBodyEarth":
                    Two body motion around Earth.
                "twoBodySun":
                    Two body motion around the Sun.
                "threeBody":
                    Three body motion.
                "threeBodySunEarth":
                    Three body motion around the Sun and Earth.
                "threeBodyEarthMoon":
                    Three body motion around the Earth and Moon.
        preset_mult (float):
            Constant multiple of potential V for 2-body motion.
            Note: Only needed if preset = "".
        variational_order (int):
            Order of variational equations to be computed:
                * 0 - for no variational equations.
                * 1 - for first order variational equations.
                * 2 - for first and second order variational equations.

    Attributes:
        vars (1-d sympy matrix):
            The variables used in the symbolic integration.
        dynamics (sympy expression):
            The dynamics to be symbolically integrated.
        variational_order (int):
            Order of variational equations to be computed:
                    * 0 - for no variational equations
                    * 1 - for first order variational equations
                    * 2 - for first and second order variational equations

        lambda_dynamics (lambdafied sympy expression):
            The lambdified dynamic equations.
        lambda_dynamics_and_variational (lambdafied sympy expression):
            The lambdified dynamic and variational equations.
        jacobian (nd sympy matrix or None):
            The jacobian of the dynamics.
        STM (nxn sympy matrix):
            The symbolic state transition matrix
    """

    def __init__(
        self, vars=None, dynamics=None, preset="", preset_mult=1, variational_order=1
    ):
        """
        Args:
            vars (1-dimensional sympy matrix):
                The variables used in the symbolic integration.
            dynamics (sympy expression(s)):
                The dynamics to be symbolically integrated.
            preset (string):
                Dynamic and Variational equation preset. Current presets are:
                    "twoBody":
                        Two body motion.
                    "twoBodyEarth":
                        Two body motion around Earth.
                    "twoBodySun":
                        Two body motion around the Sun.
                    "threeBody":
                        Three body motion.
                    "threeBodySunEarth":
                        Three body motion around the Sun and Earth.
                    "threeBodyEarthMoon":
                        Three body motion around the Earth and Moon.
            preset_mult (float):
                Constant multiple of potential V for 2-body motion.
            variational_order (int):
                Order of variational equations to be computed:
                    0 - for no variational equations.
                    1 - for first order variational equations.
                    2 - for first and second order variational equations.
        """
        # preset for two body motion
        if "twoBody" in preset:
            self._presetTwoBody(preset, preset_mult)
        elif "threeBody" in preset:
            self._presetThreeBody(preset, preset_mult)
        else:
            # create sympy symbols
            for elem in vars:
                elem = symbols(str(elem))

            self.vars = Matrix(vars)
            self.dynamics = dynamics

        # lambdify dynamics
        self.lambda_dynamics = lambdify(self.vars, self.dynamics, "numpy")

        # if user wants to use variational equations
        self._setVarEqs(variational_order)

    def _presetTwoBody(self, preset, preset_mult):
        """This method instanciates STMint under the preset of two body dynamics

        This method calculates two body motion dynamics with the option for
        preset constant multiples.

        Args:
            preset (string):
                Dynamic and Variational equation preset. Current presets are:
                    "twoBody":
                        Two body motion.
                    "twoBodyEarth":
                        Two body motion around Earth.
                    "twoBodySun":
                        Two body motion around the Sun.
            preset_mult (float):
                Constant multiple of potential V for 2-body motion.
        """

        x, y, z, vx, vy, vz = symbols("x,y,z,vx,vy,vz")

        if "Earth" in preset:
            V = const.GM_earth.to(u.km**3 / u.s**2).value / sqrt(
                x**2 + y**2 + z**2
            )
        elif "Sun" in preset:
            V = const.GM_sun.to(u.km**3 / u.s**2).value / sqrt(
                x**2 + y**2 + z**2
            )
        else:
            V = preset_mult / sqrt(x**2 + y**2 + z**2)

        r = Matrix([x, y, z])
        vr = Matrix([vx, vy, vz])
        dVdr = diff(V, r)
        RHS = Matrix.vstack(vr, dVdr)

        self.vars = Matrix([x, y, z, vx, vy, vz])
        self.dynamics = RHS

    def _presetThreeBody(self, preset, preset_mult):
        """This method instantiates STMint under the preset of three body
        restricted circular motion.

        This method calculates three body restricted circular motion dynamics
        with the option for a preset mass parameter.

        Args:
            preset (string):
                Dynamic and Variational equation preset. Current presets for
                three body motion are:
                    "threeBody":
                        Three body motion (Default to SunEarth).
                    "threeBodySunEarth":
                        Three body motion around the Sun and Earth.
                    "threeBodyEarthMoon":
                        Three body motion around the Earth and Moon.

            preset_mult (float):
                Mass parameter for two body motion (mu).
        """

        x, y, z, vx, vy, vz = symbols("x,y,z,vx,vy,vz")

        if "SunEarth" in preset:
            mu = const.M_earth.value / (const.M_earth + const.M_sun)
            mu = mu.value  # mass fraction for Earth-Sun system
        elif "EarthMoon" in preset:
            mu = const.M_moon.value / (const.M_earth + const.M_moon)
            mu = mu.value  # mass fraction for Earth-Sun system
        elif preset_mult != 1:
            mu = preset_mult
        else:
            mu = const.M_earth.value / (const.M_earth + const.M_sun)
            mu = mu.value  # mass fraction for Earth-Sun system

        mu1 = 1.0 - mu
        mu2 = mu

        r1 = sqrt((x + mu2) ** 2 + (y**2) + (z**2))
        r2 = sqrt((x - mu1) ** 2 + (y**2) + (z**2))

        U = -1.0 * (1.0 / 2.0 * (x**2 + y**2 + mu1 * mu2) + mu1 / r1 + mu2 / r2)

        dUdx = diff(U, x)
        dUdy = diff(U, y)
        dUdz = diff(U, z)

        RHS = Matrix(
            [
                vx,
                vy,
                vz,
                ((-1.0 * dUdx) + 2.0 * vy),
                ((-1.0 * dUdy) - 2.0 * vx),
                (-1.0 * dUdz),
            ]
        )

        self.vars = Matrix([x, y, z, vx, vy, vz])
        self.dynamics = RHS

    def _setVarEqs(self, variational_order):
        """This method creates or deletes associated varitional equations with
        given dynmaics

        This method first takes the jacobian of the dynamics, and creates a
        symbolic state transition matrix (STM). The jacobian and STM are then
        multiplied together to create the variational equations. These
        equations are then lambdified. If variation is False, all of these values
        are set to none.

        Args:
            variational_order (int):
                Order of variational equations to be computed:
                    0 - for no variational equations.
                    1 - for first order variational equations.
                    2 - for first and second order variational equations.
        """
        if variational_order == 1 or variational_order == 2:
            self.jacobian = self.dynamics.jacobian(self.vars.transpose())
            self.STM = MatrixSymbol("phi", len(self.vars), len(self.vars))
            self.variational = self.jacobian * self.STM
            self.lambda_dynamics_and_variational = lambdify(
                (self.vars, self.STM),
                Matrix.vstack(self.dynamics.transpose(), Matrix(self.variational)),
                "numpy",
            )
            if variational_order == 2:
                # contract the hessian to get rid of spurious dimensions
                # using sympy matrices to calculate derivative
                hessian = tensorcontraction(
                    Array(self.dynamics).diff(Array(self.vars), Array(self.vars)),
                    (1, 3, 5),
                )
                lambda_hessian = lambdify(self.vars, hessian, "numpy")

                self.jacobian = self.dynamics.jacobian(self.vars.transpose())
                lambda_jacobian = lambdify(self.vars, self.jacobian, "numpy")

                lambda_dyn = lambdify(self.vars, self.dynamics, "numpy")
                n = len(self.vars)
                self.lambda_dynamics_and_variational2 = (
                    lambda t, states: self.secondVariationalEquations(
                        lambda_dyn, lambda_jacobian, lambda_hessian, states, n
                    )
                )

        else:
            self.jacobian = None
            self.variational = None
            self.STM = None
            self.lambda_dynamics_and_variational = None

    def secondVariationalEquations(
        self, lambda_dyn, lambda_jacobian, lambda_hessian, states, n
    ):
        """This method creates the second order variational equations for given
        dynamics

        This method begins by unpacking the initial state, state transition matrix
        (STM), and state transition tensor (STT). Then, the jacobian of the state,
        time derivative of the state, STM, and STT are calculated. Finally, these
        are all returned in a single matrix.

        Agrs:
            lambda_dyn (Lambdafied function):
                Lambdafied dynamics.
            lambda_jacobian (Lambdafied function):
                Lambdafied jacobian of the dynamics.
            lambda_hessian (Lambdafied function):
                Lambdafied hessian.
            states (np array):
                Initial state of dynamics.
            n (int):
                Dimension of variables.

        Returns:
            Second order variational equations
        """
        # unpack states into three components
        state = states[:n]
        stm = np.reshape(states[n : n * (n + 1)], (n, n))
        stt = np.reshape(states[n * (n + 1) :], (n, n, n))

        # time derivative of the various components of the augmented state vector
        jac = lambda_jacobian(*state)
        dt_state = lambda_dyn(*state)
        dt_stm = np.reshape(np.matmul(jac, stm), (n**2))
        dt_stt = np.reshape(
            np.einsum("il,ljk->ijk", jac, stt)
            + np.einsum("lmi,lj,mk->ijk", lambda_hessian(*state), stm, stm),
            (n**3),
        )

        return np.hstack((dt_state.flatten(), dt_stm, dt_stt))

    # ======================================================================================================================
    # IVP Solver Functions
    # ======================================================================================================================

    def _dynamicsSolver(self, t, y):
        """Function to mimic right hand side of a dynamic system for integration

        Method unpacks initial conditions y from solve_ivp and sends it to the
        predefined lambdified dynamics.

        Args:
            t (float):
                Independent variable of initial conditions.
            y (float n array):
                Array of initial conditions of scipy.solve_ivp.

        Returns:
            lambda_dynamics (float n array):
                Array of values of dynamics subjected to initial conditions
        """
        lambda_dynamics = self.lambda_dynamics(*y).flatten()

        return lambda_dynamics

    def _dynamicsVariationalSolver(self, t, y):
        """Function to mimic right hand side of a dynamic system with variational
            equations integration

        Method unpacks initial conditions y from solve_ivp and sends it to the
        predefined lambdified dynamics and variational equations.

        Args:
            t (float):
                Independent variable of initial conditions.
            y (float n array):
                Array of initial conditions of scipy.solve_ivp.

        Returns:
            lambda_dynamics_and_variational (float n array):
                Array of values of dynamics and variational equations subjected
                to initial conditions.
        """

        l = len(self.vars)
        lambda_dynamics_and_variational = self.lambda_dynamics_and_variational(
            y[:l], np.reshape(y[l:], (l, l))
        ).flatten()

        return lambda_dynamics_and_variational

    # ======================================================================================================================
    # Clones of solve_ivp
    # ======================================================================================================================

    def dyn_int(
        self,
        t_span,
        y0,
        method="DOP853",
        t_eval=None,
        dense_output=False,
        events=None,
        vectorized=False,
        args=None,
        **options
    ):
        """Clone of scipy.solve_ivp

        Method uses _dynamicsSolver to solve an initial value problem with given
        dynamics. This method has the same arguments and Scipy's solve_ivp function.

        Non-optional arguments are listed below.
        See documentation of solve_ivp for a full list and description of arguments
        and returns
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        Args:
            t_span (2-tuple of floats):
                Interval of integration (t0, tf). The solver starts with t=t0
                and integrates until it reaches t=tf.
            y0 (array_like, shape (n,)):
                Initial state. For problems in the complex domain, pass y0 with
                a complex data type (even if the initial value is purely real).

        Returns:
            Bunch object with multiple defined fields:
                t (ndarray, shape (n_points,)):
                    Time points.
                y (ndarray, shape (n, n_points)):
                    Values of the solution at t.
                sol (OdeSolution or None):
                    Found solution as OdeSolution instance;
                    None if dense_output was set to False.
        """

        return solve_ivp(
            self._dynamicsSolver,
            t_span,
            y0,
            method,
            t_eval,
            dense_output,
            events,
            vectorized,
            args,
            **options
        )

    def dynVar_int(
        self,
        t_span,
        y0,
        output="raw",
        method="DOP853",
        t_eval=None,
        dense_output=False,
        events=None,
        vectorized=False,
        args=None,
        **options
    ):
        """Clone of scipy.solve_ivp

        Method uses _dynamicsVariationalSolver to solve an initial value
        problem with given dynamics and variational equations. This method has
        the same arguments and Scipy's solve_ivp function.

        Non-optional arguments are listed below.
        See documentation of solve_ivp for a full list and description of arguments
        and returns
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        Args:
            t_span (2-tuple of floats):
                Interval of integration (t0, tf). The solver starts with t=t0
                and integrates until it reaches t=tf.
            y0 (array_like, shape (n,)):
                Initial state. For problems in the complex domain, pass y0 with
                a complex data type (even if the initial value is purely real).
            output (str):
                Output of dynVar_int, options include:
                    * "raw":
                        Raw bunch object from scipy.solve_ivp.
                    * "final":
                        The state vector and STM at the final time only.
                    * "all":
                        The state vector and STM at all times.

        Returns:
            varies:
            If output is 'raw':
                * Bunch object with multiple defined fields, such as:
                    t (ndarray, shape (n_points,)):
                        Time points.
                    y (ndarray, shape (n, n_points)):
                        Values of the solution at t.
                    sol (OdeSolution or None):
                        Found solution as OdeSolution instance;
                        None if dense_output was set to False.
            If output is 'final':
                state (n-array):
                    The state vector.
                stm (ndarray):
                    The state transition matrix.
            If output is 'all':
                states (n-array):
                    The state vectors.
                stms (ndarray):
                    The state transition matricies.
                ts (n-array):
                    The time steps of integration.

        """
        assert self.variational != None, "Variational equations have not been created"
        initCon = np.vstack((np.array(y0), np.eye(len(self.vars))))

        solution = solve_ivp(
            self._dynamicsVariationalSolver,
            t_span,
            initCon.flatten(),
            method,
            t_eval,
            dense_output,
            events,
            vectorized,
            args,
            **options
        )

        l = len(self.vars)
        if "raw" in output:
            return solution
        if "final" in output:
            t_f = []

            for i in range(len(solution.y)):
                t_f.append(solution.y[i][-1])

            state = np.array(t_f[:l])
            stm = np.reshape(t_f[l:], (l, l))

            return state, stm

        if "all" in output:
            states = []
            stms = []
            l = len(self.vars)
            for i in range(len(solution.y[0])):
                stm = []
                state = []

                for j in range(len(solution.y)):
                    if j < l:
                        state.append(solution.y[j][i])
                    else:
                        stm.append(solution.y[j][i])

                states.append(state)
                stms.append(np.reshape(stm, (l, l)))
                ts = solution.t

            return np.array(states), stms, ts

    def dynVar_int2(
        self,
        t_span,
        y0,
        output="raw",
        method="DOP853",
        t_eval=None,
        dense_output=False,
        events=None,
        vectorized=False,
        args=None,
        **options
    ):
        """Clone of scipy.solve_ivp

        Method uses _dynamicsVariationalSolver to solve an initial value
        problem with given dynamics and variational equations. This method has
        the same arguments and Scipy's solve_ivp function. Note that this method
        also integrates second order variational equations to obtain a second
        order state transition tensor.

        Non-optional arguments are listed below.
        See documentation of solve_ivp for a full list and description of arguments
        and returns
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        Args:
            t_span (2-tuple of floats):
                Interval of integration (t0, tf). The solver starts with t=t0
                and integrates until it reaches t=tf.
            y0 (array_like, shape (n,)):
                Initial state. For problems in the complex domain, pass y0 with
                a complex data type (even if the initial value is purely real).
            output (str):
                Output of dynVar_int, options include:
                    "raw":
                        Raw bunch object from solve_ivp.
                    "final":
                        The state vector, STM, and STT at the final time only.
                    "all":
                        The state vector, STM, and STT at all times.

        Returns:
            varies:
            If output is 'raw':
                Bunch object with multiple defined fields, such as:
                    t (ndarray, shape (n_points,)):
                        Time points.
                    y (ndarray, shape (n, n_points)):
                        Values of the solution at t.
                    sol (OdeSolution or None):
                        Found solution as OdeSolution instance;
                        None if dense_output was set to False.
            If output is 'final':
                state (n-array):
                    The state vector.
                stm (ndarray):
                    The state transition matrix.
                stt (ndarray):
                    The state transition tensor.
            If output is 'all':
                states (n-array):
                    The state vectors.
                stms (ndarray):
                    The state transition matricies.
                stts (ndarray):
                    The state transition tensors.
                ts (n-array):
                    The time steps of integration.
        """
        assert self.variational != None, "Variational equations have not been created"
        init_con = np.hstack(
            (
                np.array(y0),
                np.eye(len(self.vars)).flatten(),
                np.zeros(len(self.vars) ** 3),
            )
        )

        solution = solve_ivp(
            self.lambda_dynamics_and_variational2,
            t_span,
            init_con,
            method,
            t_eval,
            dense_output,
            events,
            vectorized,
            args,
            **options
        )

        l = len(self.vars)
        if "raw" in output:
            return solution
        if "final" in output:
            t_f = []

            for i in range(len(solution.y)):
                t_f.append(solution.y[i][-1])

            state = np.array(t_f[:l])
            stm = np.reshape(t_f[l : l * (l + 1)], (l, l))
            stt = np.reshape(t_f[l * (l + 1) :], (l, l, l))

            return state, stm, stt
        if "all" in output:
            states = []
            stms = []
            stts = []

            for i in range(len(solution.y[0])):
                state = []
                stm = []
                stt = []

                for j in range(len(solution.y)):
                    if j < l:
                        state.append(solution.y[j][i])
                    elif (j >= l) and (j < (l * (l + 1))):
                        stm.append(solution.y[j][i])
                    else:
                        stt.append(solution.y[j][i])

                states.append(state)
                stms.append(np.reshape(stm, (l, l)))
                stts.append(np.reshape(stt, (l, l, l)))
                ts = solution.t

            return states, stms, stts, ts
