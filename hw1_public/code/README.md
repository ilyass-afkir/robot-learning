# Robot Learning Exercise
-------------------------
**Prepare the environment**
  1. If conda is not yet installed on your PC, you can check here: https://www.anaconda.com/products/distribution
  2. Install the conda environment:
    ```
    make install
    ```
  3. Activate the conda environment:
    ```
    conda activate robot-learning-ex1
    ```
  4. Execute examples:
    ```
    python example.py 
    ```

**You can also use "pip" to install the requirements**

**How to use the exercise latex template:**
  1. You find the template as *.zip file in the homework directory (keep it as zip):
    ```
    ../hw1_public/exercise_template.zip
    ```
  2. Open your overleaf:
    ```
    https://sharelatex01.ca.hrz.tu-darmstadt.de/project
    ```
  3. Choose the template:
    ```
    new project >  upload project > select a. zip file 
    ```
  4. Now you are able to compile it directly

**Additional Information:**
  * when the robot has to reach a fixed position with a controller in the joint space, you can execute "example.py" with:
    ```
    jointCtlComp(['P'], True), 
    jointCtlComp(['PD'], True),
    jointCtlComp(['PID'], True),
    jointCtlComp(['PD_Grav'], True) or
    jointCtlComp(['ModelBased'], True)
    ```
  * when the robot has to follow a fixed trajectory with a controller in the joint space, you can execute "example.py" with:
    ```
    jointCtlComp(['P'], False), 
    jointCtlComp(['PD'], False),
    jointCtlComp(['PID'], False),
    jointCtlComp(['PD_Grav'], False) or
    jointCtlComp(['ModelBased'], False)
    ```
  * to execute controller with the null-space task method:
    ```
    taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, pi]).T) and/or
    taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, -pi]).T)
    ```
  * by executing "example.py" with these setups, you will see the results
  * if you would like to modify the plot's setup, you can modify it in "jointCtlComp.py", function "traj_plot".
  * For this robot, the mass matrix, Coriolis and gravity are already predefined in "DoubleLink.py", and the history data are only needed for the plotting.
