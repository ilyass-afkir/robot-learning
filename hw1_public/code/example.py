from jointCtlComp import *
from taskCtlComp import *

# TASK 1.2 B: Controller in the joint space. The robot has to reach a fixed position.
# jointCtlComp(['P'], True)  # P, PD, PID, PD_Grav, ModelBased
# jointCtlComp(['PD'], True)
jointCtlComp(['PID'], True)
# jointCtlComp(['PD_Grav'], True)
# jointCtlComp(['ModelBased'], True)

# TASK 1.2 C: Same controllers in joint space as before. ONLY DIFFERENCE: The robot now has to follow a fixed trajectory
# jointCtlComp(['P'], False)  # P, PD, PID, PD_Grav, ModelBased
# jointCtlComp(['PD'], False)
# jointCtlComp(['PID'], False)
# jointCtlComp(['PD_Grav'], False)
# jointCtlComp(['ModelBased'], False)

# TASK 1.2 D: Modified version of PD controller (higher gains), but still follow a trajectory:
# jointCtlComp(['PD_high_gains'], False)

# TASK 1.2 E: Controller in the task space.
#taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, pi]).T)
# taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, -pi]).T)

input('Press Enter to close')
