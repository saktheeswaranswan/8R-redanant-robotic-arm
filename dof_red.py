import numpy as np
import matplotlib.pyplot as plt

# Sampling period
dt = 0.001  # 1 msec

# Duration of motion (in secs)
Tf = 1.0  # 1 sec
t = np.arange(0, Tf+dt, dt)

# Initial position of end-effector
pd8_0 = np.array([7, 1, 0])

xde_0 = pd8_0[0]
yde_0 = pd8_0[1]

# Final position
d_AB = 1.5
pd8_f = pd8_0 + np.array([d_AB, 0, 0])

xde_f = pd8_f[0]
yde_f = pd8_f[1]

# Desired trajectory (1st subtask): position (xd,yd); velocity (xd_,yd_)
Nmax = int(Tf/dt) + 1
k = np.linspace(0, Tf, Nmax)

a1 = 6*(xde_f - xde_0)/Tf**2
a2 = -a1/Tf

xd = (-a1*k**3)/(3*Tf) + (a1*k**2)/2 + xde_0
yd = np.ones_like(k)*1

xd_ = a1*k + a2*k**2
yd_ = np.zeros_like(k)

# Figure for animation (moving robot stick diagram)
plt.figure()
plt.axis([-1, 7, -0.5, 3])
plt.axis('on')
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
# plt.plot(xd, yd, 'm:')

dtk = 10  # Interval between consecutive plots
pauseTime = 0.1

# Kinematic simulation - main loop
print('Kinematic Simulation...')

tt = 0
tk = 1
qd = np.zeros((len(t), 8))

while tt <= Tf:
    # Calculate Jacobian for current q
    c1 = np.cos(qd[tk,0])
    c12 = np.cos(qd[tk,0] + qd[tk,1])
    c123 = np.cos(qd[tk,0] + qd[tk,1] + qd[tk,2])
    c1234 = np.cos(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3])
    c12345 = np.cos(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4])
    c123456 = np.cos(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4] + qd[tk,5])
    c1234567 = np.cos(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4] + qd[tk,5] + qd[tk,6])
    c12345678 = np.cos(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4] + qd[tk,5] + qd[tk,6] + qd[tk,7])
    s1 = np.sin(qd[tk,0])
    s12 = np.sin(qd[tk,0] + qd[tk,1])
    s123 = np.sin(qd[tk,0] + qd[tk,1] + qd[tk,2])
    s1234 = np.sin(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3])
    s12345 = np.sin(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4])
    s123456 = np.sin(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4] + qd[tk,5])
    s1234567 = np.sin(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4] + qd[tk,5] + qd[tk,6])
    s12345678 = np.sin(qd[tk,0] + qd[tk,1] + qd[tk,2] + qd[tk,3] + qd[tk,4] + qd[tk,5] + qd[tk,6] + qd[tk,7])
    
    Jac1 = np.zeros((2, 8))
    Jac1[0, 0] = -c1 - s12 - c123 - s1234 - s12345 - s123456 + c1234567 - s12345678
    Jac1[0, 1] = Jac1[0, 0] + c1
    Jac1[0, 2] = Jac1[0, 1] + s12
    Jac1[0, 3] = Jac1[0, 2] + c123
    Jac1[0, 4] = Jac1[0, 3] + s1234
    Jac1[0, 5] = Jac1[0, 4] + s12345
    Jac1[0, 6] = c1234567 - s12345678
    Jac1[0, 7] = -s12345678
    
    Jac1[1, 0] = -s1 + c12 - s123 + c1234 + c12345 + c123456 + s1234567 + c12345678
    Jac1[1, 1] = Jac1[1, 0] + s1
    Jac1[1, 2] = Jac1[1, 1] - c12
    Jac1[1, 3] = Jac1[1, 2] + s123
    Jac1[1, 4] = Jac1[1, 3] - c1234
    Jac1[1, 5] = c12345678 + c123456 + s1234567
    Jac1[1, 6] = c12345678 + s1234567
    Jac1[1, 7] = c12345678
    
    # Pseudo-inverse of jacobian
    Jac1_psinv = np.dot(np.transpose(Jac1), np.linalg.inv(np.dot(Jac1, np.transpose(Jac1))))
    
    # Subtask 1
    xd_ = 0
    yd_ = 0
    task1 = np.dot(Jac1_psinv, np.array([[xd_], [yd_]]))
    
    # Forward kinematics
    xd1 = -s1
    yd1 = c1
    
    xd2 = xd1 + c12
    yd2 = yd1 + s12
    
    xd3 = xd2 - s123
    yd3 = yd2 + c123
    
    xd4 = xd3 + c1234
    yd4 = yd3 + s1234

    xd5 = xd4 + c12345
    yd5 = yd4 + s12345
    
    xd6 = xd5 + c123456
    yd6 = yd5 + s123456
    
    xd7 = xd6 + s1234567
    yd7 = yd6 - c1234567
    
    xd8 = xd7 + c12345678
    yd8 = yd7 + s12345678

    dist7_m = (yd7 -yde_0)

    xi7 = np.zeros((8, 1))
    xi7[0] = -2*dist7_m*(-s1 + c12 - s123 + c1234 + c12345 + c123456 + s1234567)
    xi7[1] = -2*dist7_m*(c12 - s123 + c1234 + c12345 + c123456 + s1234567)
    xi7[2] = -2*dist7_m*(-s123 + c1234 + c12345 + c123456 + s1234567)
    xi7[3] = -2*dist7_m*(c1234 + c12345 + c123456 + s1234567)
    xi7[4] = -2*dist7_m*(c12345 + c123456 + s1234567)
    xi7[5] = -2*dist7_m*(c123456 + s1234567)
    xi7[6] = -2*dist7_m*(s1234567)
    xi7[7] = -2*dist7_m*0

    dist7_m = dist7_m**2

    dist6_m = (yd6 -yde_0)

    xi6 = np.zeros((8, 1))
    xi6[0] = -2*dist6_m*(-s1 + c12 - s123 + c1234 + c12345 + c123456)
    xi6[1] = -2*dist6_m*(c12 - s123 + c1234 + c12345 + c123456)      
    xi6[2] = -2*dist6_m*(-s123 + c1234 + c12345 + c123456)           
    xi6[3] = -2*dist6_m*(c1234 + c12345 + c123456)                   
    xi6[4] = -2*dist6_m*(c12345 + c123456)                           
    xi6[5] = -2*dist6_m*(c123456)                                    
    xi6[6] = -2*dist6_m*0
    xi6[7] = -2*dist6_m*0

    dist6_m = dist6_m**2

    dist5_m = (yd5 -yde_0)**2

    xi5 = np.zeros((8, 1))
    xi5[0] = -2*dist5_m*(-s1 + c12 - s123 + c1234 + c12345)
    xi5[1] = -2*dist5_m*(c12 - s123 + c1234 + c12345)     
    xi5[2] = -2*dist5_m*(-s123 + c1234 + c12345)           
    xi5[3] = -2*dist5_m*(c1234 + c12345)                  
    xi5[4] = -2*dist5_m*(c12345)                          
    xi5[5] = -2*dist5_m*0
    xi5[6] = -2*dist5_m*0
    xi5[7] = -2*dist5_m*0

    dist5_m = dist5_m**2

    k5 = 50
    k6 = 50
    k7 = 100
    task2 = (np.identity(8) - Jac1_psinv @ Jac1) @ (k5 * xi5 + k6 * xi6 + k7 * xi7)

    if max([dist5_m, dist6_m, dist7_m]) >= 0.01:
        qd_ = np.transpose(task2)
    else:
        qd_ = task1 + np.transpose(task2)

    # End-effector velocity
    v8 = Jac1 @ np.transpose(qd_)

    # Numerical integration --> Kinematic simulation
    # print("qd_ = ", qd_[0,0], "qd = ", qd[tk, :], "qd[tk+1, :] = ", qd[tk+1, :], "v8 = ", v8)
    qd[tk+1, 0] = qd[tk, 0] + dt * qd_[0,0]
    qd[tk+1, 1] = qd[tk, 1] + dt * qd_[0,1]
    qd[tk+1, 2] = qd[tk, 2] + dt * qd_[0,2]
    qd[tk+1, 3] = qd[tk, 3] + dt * qd_[0,3]
    qd[tk+1, 4] = qd[tk, 4] + dt * qd_[0,4]
    qd[tk+1, 5] = qd[tk, 5] + dt * qd_[0,5]
    qd[tk+1, 6] = qd[tk, 6] + dt * qd_[0,6]
    qd[tk+1, 7] = qd[tk, 7] + dt * qd_[0,7]

    #plot robot
    if (tk % dtk == 0 or tk == 1):
        plt.clf() #clear plot
        # plt.plot(xd, yd, 'm:')
        
        # plt.plot(0, 0, 'o')
        
        # plt.plot([0, xd1[tk]], [0, yd1[tk]],'b')
        # plt.plot(xd1[tk], yd1[tk], 'ro')
        
        # plt.plot([xd1[tk], xd2[tk]], [yd1[tk], yd2[tk]],'b')
        # plt.plot(xd2[tk], yd2[tk], 'ro')
        
        # plt.plot([xd2[tk], xd3[tk]], [yd2[tk], yd3[tk]],'b')
        # plt.plot(xd3[tk], yd3[tk], 'ro')
        
        # plt.plot([xd3[tk], xd4[tk]], [yd3[tk], yd4[tk]],'b')
        # plt.plot(xd4[tk], yd4[tk], 'ro')
        
        # plt.plot([xd4[tk], xd5[tk]], [yd4[tk], yd5[tk]],'b')
        # plt.plot(xd5[tk], yd5[tk], 'ro')
        
        # plt.plot([xd5[tk], xd6[tk]], [yd5[tk], yd6[tk]],'b')
        # plt.plot(xd6[tk], yd6[tk], 'ro')
        
        # plt.plot([xd6[tk], xd7[tk]], [yd6[tk], yd7[tk]],'b')
        # plt.plot(xd7[tk], yd7[tk], 'ro')
        
        # plt.plot([xd7[tk], xd8[tk]], [yd7[tk], yd8[tk]],'b')
        # plt.plot(xd8[tk], yd8[tk], 'ro')
        
        # plt.plot(xd8[tk], yd8[tk], 'g+')

        print(tk)

        plt.plot([0, xd1,xd2 ,xd3 ,xd4 ,xd5 ,xd6 ,xd7 ,xd8 ], [0, yd1 , yd2 , yd3 , yd4 , yd5 , yd6 , yd7 , yd8 ], 'ro-', lw=3, markersize=8)
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.title('8R Manipulator Stick Diagram')
        
        #OBSTACLES
        diam = 0.4
        d = 1
        plt.gca().add_patch(plt.Rectangle((xde_0 - diam/2, yde_0 - d/2-diam), diam, diam, linewidth=1, edgecolor='black', facecolor='blue'))
        plt.gca().add_patch(plt.Rectangle((xde_0 - diam/2, yde_0 + d/2), diam, diam, linewidth=1, edgecolor='black', facecolor='blue'))
        plt.show()

    tk = tk + 1
    tt = tt + dt
