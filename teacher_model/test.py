# To estimate shuttle height from a single video frame, you skip estimateAffine3D and use Single View Metrology based on the court’s known geometry. Here is the workflow:
# Map the Court (Calibration): Use cv2.solvePnP. You input the 2D pixels of the 4 court corners and their real-world 3D coordinates (e.g., 
# , 
# ). This tells the computer exactly where your camera is in 3D space.
# Locate the Ground Point: Identify the 2D pixel directly under the shuttle on the floor. If the shuttle is over a line (like the service line), this is easy. If not, you must estimate its "shadow" position based on the court grid.
# Establish Vertical Scale: Use the net pole (1.55m) as a reference. Because of perspective, a pixel at the net is "worth" more meters than a pixel at the back of the court.
# Calculate Perspective Height: Use the camera's vanishing points (calculated from the parallel court lines) to "transfer" the scale from the net pole to the shuttle's specific distance.
# Final Calculation: Measure the vertical pixel distance between the shuttle and its ground point, then multiply by your distance-corrected "meters-per-pixel" ratio to get the Z (height).
# The catch: Accuracy depends entirely on how correctly you identify the "ground point" under the floating shuttle.
# Do you want the formula or Python code to calculate the distance-corrected scale using the vanishing points?



# can i use the equation of trajectory of the shuttle
# Yes, using the trajectory equation is actually the most "pro" way to solve this. In fact, it's how many automated sports tracking systems (like Hawk-Eye) handle single-camera data.
# By using the physics of a shuttlecock, you solve the "Ground Point" problem because the trajectory dictates exactly where the shuttle must be in 3D space to satisfy both the image and the laws of physics.
# How to combine Physics with your OpenCV approach:
# 1. The Parabolic (or Drag) Model
# A shuttlecock doesn't follow a perfect parabola because of high air resistance (drag). It follows a distorted parabola (steeper on the way down). The equation looks like this:

# However, for a simple estimate, you can use a standard projectile motion equation and solve for the unknown height 
#  over time 
# .
# 2. The "Ray-Trajectory" Intersection
# This is the "magic" step:
# For every frame in your video, the camera sees the shuttle at a 2D pixel 
# .
# Using your solvePnP camera calibration, that 2D pixel forms a 3D ray (a line) extending from the camera lens out into the stadium.
# The shuttle must exist somewhere along that ray.
# When you look at multiple frames, you get multiple rays. There is only one unique 3D trajectory curve that passes through all those rays while following the laws of gravity.
# Why this is better than "Option A":
# No "Ground Point" Needed: You don't need to guess where the shuttle's shadow is. The math finds the 3D curve that fits the sequence of dots.
# Smoothes Out Errors: If you click slightly off on one frame, the physics model "ignores" the noise and keeps the path smooth.
# Predicts the Land: It allows you to predict exactly where the shuttle will land before it hits the floor.
# The Workflow
# Track the shuttle in 2D for several frames (at least 5-10).
# Calibrate the court using solvePnP to get your camera's 3D position.
# Perform "Best Fit": Use a mathematical solver (like scipy.optimize) to find a 3D trajectory that, when projected back onto a 2D screen, matches your video pixels as closely as possible.
# Would you like a simplified version of the math (the projectile motion equations) to start building your trajectory model?
# In biomechanics, the point of contact in a racket sport perfectly correlates with Peak Wrist Velocity or Peak Extension Angle.
# When YOLO loses the shuttle, you just look at the 30 frames of data your BiLSTM already extracted and find the frame where the racket hand is moving the fastest:
# $T_{impact} = \arg\max_{t \in [0, 30]} (V_{wrist}(t))$




# Refined Data Pipeline StructureScript 1: The "Auto-Labeler" (Hit Detector)Input: Video + MediaPipe CSV.Output: Updated CSV with a hit_frame column (1/0).Value: 
# This creates the "Temporal Anchors" for your physics. You now know exactly when the $Z$ coordinate is at its peak (Smash) or at the floor.
# Script 2: The "Vision Pass" (TrackNetV3)Input: Video.Output: Shuttle image_x, image_y for every frame.Value: This provides the "Observed Path." 
# Even if it’s noisy, it’s better than pure simulation.Script 3: The "Spatial Resolver" (SolvePnP + Correction)Task A: Run solvePnP once per video to get the Camera 
# Matrix.Task B: Use the "Similar Triangles" formula with the Camera $Z$ from SolvePnP to project the TrackNet pixels onto the court.
# Task C: Calculate the final $Z$ using the time-of-flight between the hit_frame markers from Script 1.




