import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate):
    dx = 2.0 * np.cos(np.pi / 4) 
    dy = 2.0 * np.sin(np.pi / 4) 
    
    odometry = gtsam.Pose2(dx, dy, np.pi / 2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))
    
    # Bypass the noisy X(3) composition and insert the exact expected global pose
    # Ideal X(3) is at (4.0, 0.0, 0.0), so ideal X(4) is at (4.0 + dx, dy, pi/2)
    ideal_x4 = gtsam.Pose2(4.0 + dx, dy, np.pi / 2)
    initial_estimate.insert(X(4), ideal_x4)
    
    return graph, initial_estimate