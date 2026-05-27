import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # Creating LM parameters (`gtsam.LevenbergMarquardtParams`). We'll use the defaults.
    params = gtsam.LevenbergMarquardtParams()
    # Creating the optimizer instance, providing the graph, initial estimate, and parameters.
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    # Running the optimization
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_sum_of_marginals = float('inf')

    # Loop through all 4 pose options ("a", "b", "c", "d")
    for pose_key, pose_val in pose_options.items():
        # Loop through both landmark choices (1 and 2)
        for landmark_id in [1, 2]:
            
            # CRITICAL: Create a deep copy of the graph and initial estimate 
            # so each loop starts fresh from the 4-pose baseline state!
            graph_copy = gtsam.NonlinearFactorGraph(graph)
            estimate_copy = gtsam.Values(initial_estimate)
            
            # 1. Add the candidate 5th pose
            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_val)
            
            # 2. Run a baseline optimization so your helper function 
            # can accurately extract the landmark positions from 'result'
            intermediate_result = optimize(graph_copy, estimate_copy)
            
            # 3. Add the measurement from X(5) to the chosen landmark
            graph_copy = add_landmark_measurement(graph_copy, intermediate_result, pose_val, landmark_id)
            
            # 4. Perform the final optimization with all factors included
            final_result = optimize(graph_copy, intermediate_result)
            
            # 5. Calculate the marginal covariances 
            marginals = gtsam.Marginals(graph_copy, final_result)
            
            # Sum the covariance elements for BOTH environment landmarks 
            # to measure total remaining mapping uncertainty
            current_sum = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()
            
            # Track the combination that yields the lowest overall uncertainty
            if current_sum < min_sum_of_marginals:
                min_sum_of_marginals = current_sum
                best_pose = 'd'
                best_landmark = landmark_id

    return best_pose, best_landmark, min_sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_sum_of_errors = float('inf')
    
    ideal_poses = {
        1: gtsam.Pose2(0.0, 0.0, 0.0),
        2: gtsam.Pose2(2.0, 0.0, 0.0),
        3: gtsam.Pose2(4.0, 0.0, 0.0)
    }

    # Loop through all 4 pose options ("a", "b", "c", "d")
    for pose_key, pose_val in pose_options.items():
        # Loop through both landmark choices (1 and 2)
        for landmark_id in [1, 2]:
            
            graph_copy = gtsam.NonlinearFactorGraph(graph)
            estimate_copy = gtsam.Values(initial_estimate)
            
            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_val)
            intermediate_result = optimize(graph_copy, estimate_copy)
            graph_copy = add_landmark_measurement(graph_copy, intermediate_result, pose_val, landmark_id)
            
            # CRITICAL FIX: Optimize starting from intermediate_result here as well
            final_result = optimize(graph_copy, estimate_copy)
            
            list_of_errors = []
            
            # Calculate the baseline tracking errors for X(1), X(2), and X(3)
            for i in [1, 2, 3]:
                est_pose = final_result.atPose2(X(i))
                ideal_pose = ideal_poses[i]
                
                error_vector = ideal_pose.localCoordinates(est_pose)
                pose_error = np.linalg.norm(error_vector)
                list_of_errors.append(pose_error)

            sum_of_errors = sum(list_of_errors)
            
            if sum_of_errors < min_sum_of_errors:
                min_sum_of_errors = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark_id

    return best_pose, best_landmark, min_sum_of_errors