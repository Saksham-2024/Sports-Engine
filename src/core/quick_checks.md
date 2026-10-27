1. make sure the file paths are either using the os library or the Pathlib library
2. Make sure that you are correctly using get_path() and get_config() methods from the configs class into your other modules while creating objects and resolving
3. verify configs are correctly mentioned in the configs.yaml file and correctly loaded in each file
4. verify each file has proper logging
5. ensure correct paths are being used in each module
5. lint all files for bugs
7. ensure solid design principles
8. in the calieration pipeline classs inside run method, we check for if video doesnt have homography adn then get points and then if it doesnt have camera pos, we again ask users to click on points. this is redundant and frustrating. fix this (camera_matrice.py)
9. keep in mind that default start frame in case no segments are available before homography/camera_pos calculation is 7500 - according to BWF videos. actual videos from real users may be short and we might kip through the entire video or most of 