from __future__ import (absolute_import, division, unicode_literals)
import numpy as np
from numpy import dot, zeros, eye


class CollaborativeKalmanFilter(object):

    """ dim_a - extra agents besides the current one
        agent_id - 0...dim_a-1 
    """

    def __init__(self, dim_x, dim_z, dim_a, agent_id, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.dim_a = dim_a

        self.aid = agent_id

        self.x = zeros((dim_x, 1))  # state
        self.P = eye(dim_x)        # uncertainty covariance
        # cross covariance of agents
        self.cP = [eye(dim_x) for _ in range(dim_a)]  # TODO: configurable
        self.B = 0                 # control transition matrix
        self.F = np.eye(dim_x)     # state transition matrix
        self.R = eye(dim_z)        # state uncertainty
        self.rR = eye(1)           # TODO:
        self.Q = eye(dim_x)        # process uncertainty
        self.y = zeros((dim_z, 1))  # residual

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape)  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        """ Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, posterior is not computed

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, *args)

        # check for inf in H
        if np.isinf(H).any():
            raise ValueError("H contains inf")
        # check for nan in H
        if np.isnan(H).any():
            raise ValueError("H contains nan")

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.K = PHT.dot(np.linalg.inv(self.S))

        hx = Hx(self.x, *hx_args)
        self.y = residual(z, hx)
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # save posterior state
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        for i in range(self.dim_a):
            if i == self.aid:
                continue
            self.cP[i] = I_KH @ self.cP[i]

    def rel_update(self, aid, ax, aP, aSji, z, HJacobian, Hx, R=None, args=(), hx_args=(),
                   residual=np.subtract):
        """ Relative update between two agents using collaborative extended Kalman filter."""
        Pii = self.P.copy()
        Pij = self.cP[aid].copy() @ aSji.T
        Paa = np.block([[Pii, Pij],
                        [Pij.T, aP]])

        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R
        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, ax, *args)
        Hax = HJacobian(ax, self.x, *args)
        nz = Hx(self.x, *hx_args)

        Fa = np.block([H, Hax])

        Ka = Paa @ Fa.T @ np.linalg.inv(Fa @ Paa @ Fa.T + self.rR)  # TODO:
        Xij = np.block([[self.x],
                        [ax]])
        Xij += Ka @ (z - nz)
        # update the state
        self.x = Xij[:self.dim_x].copy()
        xj = Xij[self.dim_x:]
        Paa = (np.eye(self.dim_x * 2) - Ka @ Fa) @ Paa
        self.cP[aid] = Paa[:self.dim_x, self.dim_x:].copy()  # update Sij

        Pii = Paa[:self.dim_x, :self.dim_x]
        self.P = Pii.copy()
        # enforce symmetry, solely for numerical stability
        self.P = (self.P + self.P.T)/2

        for i in range(self.dim_a):
            if i == self.aid:
                continue
            if i == aid:
                continue
            self.cP[i] = Pii @ np.linalg.inv(Pii) @ self.cP[i]

        # the rest will be outside of filter
        Pjj = Paa[self.dim_x:, self.dim_x:]
        return (xj.copy(), Pjj.copy())

    def predict_x(self, u=0):
        """
        Predicts the next state of X. If you need to
        compute the next state yourself, override this function. You would
        need to do this, for example, if the usual Taylor expansion to
        generate F is not providing accurate results for you.
        """
        self.x = dot(self.F, self.x) + dot(self.B, u)

    def predict(self, u=0):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        self.predict_x(u)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        # CKF addition
        for i in range(self.dim_a):
            if i == self.aid:
                continue
            self.cP[i] = self.F @ self.cP[i]
