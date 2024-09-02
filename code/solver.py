from tqdm import tqdm
import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import time
from scipy.linalg import norm
# from vctoolkit import Timer


class Solver:
  def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8, verbose=False):
    """
    Parameters
    ----------
    eps : float, optional
      Epsilon for derivative computation, by default 1e-5
    max_iter : int, optional
      Max iterations, by default 30
    mse_threshold : float, optional
      Early top when mse change is smaller than this threshold, by default 1e-8
    verbose : bool, optional
      Print information in each iteration, by default False
    """
    self.eps = eps
    self.max_iter = max_iter
    self.mse_threshold = mse_threshold
    self.verbose = verbose
    # self.timer = Timer()

  def get_derivative(self, model, params, n):
    """
    Compute the derivative by adding and subtracting epsilon

    Parameters
    ----------
    model : object
      Model wrapper to be manipulated.
    params : np.ndarray
      Current model parameters.
    n : int
      The index of parameter.

    Returns
    -------
    np.ndarray
      Derivative with respect to the n-th parameter.
    """
    params1 = np.array(params)
    params2 = np.array(params)

    params1[n] += self.eps
    params2[n] -= self.eps

    res1 = model.run(params1)
    res2 = model.run(params2)

    d = (res1 - res2) / (2 * self.eps)

    return d.ravel()

  def solve(self, model, target, init=None, u=1e-3, v=1.5):
    """
    Solver for the target.

    Parameters
    ----------
    model : object
      Wrapper to be manipulated.
    target : np.ndarray
      Optimization target.
    init : np,ndarray, optional
      Initial parameters, by default None
    u : float, optional
      LM algorithm parameter, by default 1e-3
    v : float, optional
      LM algorithm parameter, by default 1.5

    Returns
    -------
    np.ndarray
      Solved model parameters.
    """
    if init is None:
      init = np.zeros(model.n_params)
    out_n = np.shape(model.run(init).ravel())[0]
    jacobian = np.zeros([out_n, init.shape[0]])

    last_update = 0
    last_mse = 0
    params = init
    for i in range(self.max_iter):
      residual = (model.run(params) - target).reshape(out_n, 1)
      mse = np.mean(np.square(residual))

      if abs(mse - last_mse) < self.mse_threshold:
        return params

      for k in range(params.shape[0]):
        jacobian[:, k] = self.get_derivative(model, params, k)

      jtj = np.matmul(jacobian.T, jacobian)
      jtj = jtj + u * np.eye(jtj.shape[0])

      update = last_mse - mse
      delta = np.matmul(
        np.matmul(np.linalg.inv(jtj), jacobian.T), residual
      ).ravel()
      params -= delta

      if update > last_update and update > 0:
        u /= v
      else:
        u *= v

      last_update = update
      last_mse = mse

      if self.verbose:
        # print(i, self.timer.tic(), mse)
        print(i, mse)

    return params

import numpy as np
from scipy.optimize import least_squares

class SolverLS:
    def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8, verbose=False):
        """
        Parameters
        ----------
        eps : float, optional
          Epsilon for derivative computation, by default 1e-5
        max_iter : int, optional
          Max iterations, by default 30
        mse_threshold : float, optional
          Early stop when mse change is smaller than this threshold, by default 1e-8
        verbose : bool, optional
          Print information in each iteration, by default False
        """
        self.eps = eps
        self.max_iter = max_iter
        self.mse_threshold = mse_threshold
        self.verbose = verbose

    def solve(self, model, target, init=None):
        """
        Solver for the target.

        Parameters
        ----------
        model : object
          Wrapper to be manipulated.
        target : np.ndarray
          Optimization target.
        init : np,ndarray, optional
          Initial parameters, by default None

        Returns
        -------
        np.ndarray
          Solved model parameters.
        """
        if init is None:
            init = np.zeros(model.n_params)

        def residuals(params):
            return (model.run(params) - target).ravel()

        result = least_squares(residuals, init, xtol=self.mse_threshold, max_nfev=self.max_iter, verbose=2 if self.verbose else 0)

        if self.verbose:
            print("Optimization result:", result)

        return result.x , model.run(result.x) 
    
from scipy.optimize import minimize

class SolverBFGS:
    def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8, verbose=False):
        """
        Parameters
        ----------
        eps : float, optional
          Epsilon for derivative computation, by default 1e-5
        max_iter : int, optional
          Max iterations, by default 30
        mse_threshold : float, optional
          Early stop when mse change is smaller than this threshold, by default 1e-8
        verbose : bool, optional
          Print information in each iteration, by default False
        """
        # self.eps = eps
        # self.max_iter = max_iter
        # self.mse_threshold = mse_threshold
        self.verbose = verbose

    def solve(self, model, target, init=None):
        """
        Solver for the target.

        Parameters
        ----------
        model : object
          Wrapper to be manipulated.
        target : np.ndarray
          Optimization target.
        init : np,ndarray, optional
          Initial parameters, by default None

        Returns
        -------
        np.ndarray
          Solved model parameters.
        """
        if init is None:
            init = np.zeros(model.n_params)
            print('init is',init)
            print('model.n_params is ',model.n_params)

        def cost_function(params):
            # residual = (model.run(params) - target).ravel()
            residual = norm(model.run(params) - target) 
            # print('model shape is')
            # print(model.decode(params)[0])
            K = np.sum(residual**2)
            S = norm(model.decode(params)[0])
            # model.set_params(shape=np.zeros(10))
            return K + S
        
        def cost_function_weighted(params):
            residual = (model.run(params) - target)
            residual[:][2] = residual[:][2]*10
            residual[0][:] = residual[0][:]*10
            # residual[0][2] = residual[0][2]*10000
            residual = residual.ravel()
            return np.sum(residual**2)
        

        rr.init("rerun_example_my_data", spawn=True)  # Initialize Rerun viewer

        if target is not None:
            rr.log("real", rr.Points3D(target, colors=(0, 0, 255),labels=[str(i) for i in range(22)], radii=1))

        start_time = time.time()

        def callback(params):
            elapsed_time = time.time() - start_time
            estimated_keypoints = model.run(params)
            rr.log("estimated", rr.Points3D(estimated_keypoints, colors=(0, 255, 0),labels=[str(i) for i in range(22)], radii=1))
            if self.verbose:
                print(f"Time: {elapsed_time:.2f}s")
                print("Current parameters:", params)
                print("Current MSE:", cost_function(params))


        options = { 'disp': self.verbose,
                  #  'maxfun': 5000,
                   'gtol': 21*1e-8
                  #  'xrtol': 0
                   }
        # options = {'maxiter': self.max_iter, 'disp': self.verbose}
        result = minimize(cost_function, init, method='BFGS', options=options,callback= callback,tol = 1e-8)
                          #  tol=self.mse_threshold)
        print(result.message)

        if self.verbose:
            print("Optimization result:", result)

        return result.x , model.run(result.x) 
    
    def solveWrist(self, model, target, init=None):
        """
        Solver for the target. Trying to find the params of a hand 
        with the same keypoint 0 of the wrist that will be then used for initialization on the above solver

        Parameters
        ----------
        model : object
          Wrapper to be manipulated.
        target : np.ndarray
          Optimization target. Only the keypoint[0]
        init : np,ndarray, optional
          Initial parameters, by default None

        Returns
        -------
        np.ndarray
          Solved model parameters.
        """
        if init is None:
            init = np.zeros(model.n_params)
            # print('init is',init)
            # print('model.n_params is ',model.n_params)

        def cost_function(params):
            residual = (model.run(params)[0:4] - target[0:4]).ravel()
            return np.sum(residual**2)
        

        rr.init("rerun_example_my_data", spawn=True)  # Initialize Rerun viewer

        if target is not None:
            rr.log("real", rr.Points3D(target[0:2], colors=(0, 0, 255), radii=1))

        start_time = time.time()

        def callback(params):
            elapsed_time = time.time() - start_time
            estimated_keypoints = model.run(params)
            rr.log("estimated", rr.Points3D(estimated_keypoints[0:2], colors=(0, 255, 0), radii=1))
            if self.verbose:
                print(f"Time: {elapsed_time:.2f}s")
                print("Current parameters:", params)
                print("Current MSE:", cost_function(params))


        options = { 'disp': self.verbose}
        # options = {'maxiter': self.max_iter, 'disp': self.verbose}
        result = minimize(cost_function, init, method='BFGS', options=options,callback= callback)#, tol=self.mse_threshold)

        if self.verbose:
            print("Optimization result:", result)

        return result.x , model.run(result.x) 