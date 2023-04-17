#Kinematic Control of Redundant Manipulator

import numpy as np
import matplotlib.pyplot as plt
clear
close_('all')
#Sampling period
dt = 0.001

#duration of motion (in secs)
Tf = 1.0

t = np.arange(0,Tf+dt,dt)
#Initial position of end-effector
pd8_0 = np.array([[5],[1],[0]])
xde_0 = pd8_0(1)
yde_0 = pd8_0(2)
#final position
d_AB = 1.5
pd8_f = pd8_0 + np.array([[d_AB],[0],[0]])
xde_f = pd8_f(1)
yde_f = pd8_f(2)
# desired trajectory (1st subtask): position (xd,yd); velocity (xd_,yd_)
Nmax = Tf / dt + 1
k = np.linspace(0,Tf,Nmax)
a1 = 6 * (xde_f - xde_0) / Tf ** 2
a2 = - a1 / Tf
xd = (- a1 * k ** 3) / (3 * Tf) + (a1 * k ** 2) / 2 + xde_0
yd = np.ones((1,len(k))) * 1
xd_ = a1 * k + a2 * k ** 2
yd_ = np.zeros((1,len(k)))
########## FIGURE FOR ANIMATION ############
####### (moving robot stick diagram) #######

figure
#set(gcf, 'Position', get(0,'Screensize')); # Maximize figure.
plt.axis(np.array([- 1,7,- 0.5,3]))
plt.axis('on')
plt.axis('equal')
hold('on')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.plot(xd,yd,'m:')
dtk = 10

pauseTime = 0.1
# ****** KINEMATIC SIMULATION - Main loop ******
print('Kinematic Simulation ...')
tt = 0
tk = 1
qd[tk,1] = 0
qd[tk,2] = 0
qd[tk,3] = 0
qd[tk,4] = 0
qd[tk,5] = 0
qd[tk,6] = 0
qd[tk,7] = 0
qd[tk,8] = 0
while (tt <= Tf):

    #Calculate Jacobian for current q
    c1 = np.cos(qd(tk,1))
    c12 = np.cos(qd(tk,1) + qd(tk,2))
    c123 = np.cos(qd(tk,1) + qd(tk,2) + qd(tk,3))
    c1234 = np.cos(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4))
    c12345 = np.cos(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5))
    c123456 = np.cos(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5) + qd(tk,6))
    c1234567 = np.cos(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5) + qd(tk,6) + qd(tk,7))
    c12345678 = np.cos(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5) + qd(tk,6) + qd(tk,7) + qd(tk,8))
    s1 = np.sin(qd(tk,1))
    s12 = np.sin(qd(tk,1) + qd(tk,2))
    s123 = np.sin(qd(tk,1) + qd(tk,2) + qd(tk,3))
    s1234 = np.sin(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4))
    s12345 = np.sin(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5))
    s123456 = np.sin(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5) + qd(tk,6))
    s1234567 = np.sin(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5) + qd(tk,6) + qd(tk,7))
    s12345678 = np.sin(qd(tk,1) + qd(tk,2) + qd(tk,3) + qd(tk,4) + qd(tk,5) + qd(tk,6) + qd(tk,7) + qd(tk,8))
    Jac1[1,1] = - c1 - s12 - c123 - s1234 - s12345 - s123456 + c1234567 - s12345678
    Jac1[1,2] = Jac1(1,1) + c1
    Jac1[1,3] = Jac1(1,2) + s12
    Jac1[1,4] = Jac1(1,3) + c123
    Jac1[1,5] = Jac1(1,4) + s1234
    Jac1[1,6] = Jac1(1,5) + s12345
    Jac1[1,7] = c1234567 - s12345678
    Jac1[1,8] = - s12345678
    Jac1[2,1] = - s1 + c12 - s123 + c1234 + c12345 + c123456 + s1234567 + c12345678
    Jac1[2,2] = Jac1(2,1) + s1
    Jac1[2,3] = Jac1(2,2) - c12
    Jac1[2,4] = Jac1(2,3) + s123
    Jac1[2,5] = Jac1(2,4) - c1234
    Jac1[2,6] = c12345678 + c123456 + s1234567
    Jac1[2,7] = c12345678 + s1234567
    Jac1[2,8] = c12345678
    #Pseudo-inverse of jacobian
    Jac1_psinv = np.transpose(Jac1) / (Jac1 * np.transpose(Jac1))
    #Subtask 1
    task1 = Jac1_psinv * np.array([[xd_(tk)],[yd_(tk)]])
    #forward kinematics
    xd1[tk] = - s1
    yd1[tk] = c1
    xd2[tk] = xd1(tk) + c12
    yd2[tk] = yd1(tk) + s12
    xd3[tk] = xd2(tk) - s123
    yd3[tk] = yd2(tk) + c123
    xd4[tk] = xd3(tk) + c1234
    yd4[tk] = yd3(tk) + s1234
    xd5[tk] = xd4(tk) + c12345
    yd5[tk] = yd4(tk) + s12345
    xd6[tk] = xd5(tk) + c123456
    yd6[tk] = yd5(tk) + s123456
    xd7[tk] = xd6(tk) + s1234567
    yd7[tk] = yd6(tk) - c1234567
    xd8[tk] = xd7(tk) + c12345678
    yd8[tk] = yd7(tk) + s12345678
    #Subtask 2
#minimize distance of joints 6 and 7 from middle line between obstacles
    #joint 7
    dist7_m[tk] = (yd7(tk) - yde_0) ** 2
    xi7 = np.zeros((8,1))
    xi7[1] = - 2 * (yd7(tk) - yde_0) * (- s1 + c12 - s123 + c1234 + c12345 + c123456 + s1234567)
    xi7[2] = - 2 * (yd7(tk) - yde_0) * (c12 - s123 + c1234 + c12345 + c123456 + s1234567)
    xi7[3] = - 2 * (yd7(tk) - yde_0) * (- s123 + c1234 + c12345 + c123456 + s1234567)
    xi7[4] = - 2 * (yd7(tk) - yde_0) * (c1234 + c12345 + c123456 + s1234567)
    xi7[5] = - 2 * (yd7(tk) - yde_0) * (c12345 + c123456 + s1234567)
    xi7[6] = - 2 * (yd7(tk) - yde_0) * (c123456 + s1234567)
    xi7[7] = - 2 * (yd7(tk) - yde_0) * (s1234567)
    xi7[8] = 0
    #joint 6
    dist6_m[tk] = (yd6(tk) - yde_0) ** 2
    xi6 = np.zeros((8,1))
    xi6[1] = - 2 * (yd6(tk) - yde_0) * (- s1 + c12 - s123 + c1234 + c12345 + c123456)
    xi6[2] = - 2 * (yd6(tk) - yde_0) * (c12 - s123 + c1234 + c12345 + c123456)
    xi6[3] = - 2 * (yd6(tk) - yde_0) * (- s123 + c1234 + c12345 + c123456)
    xi6[4] = - 2 * (yd6(tk) - yde_0) * (c1234 + c12345 + c123456)
    xi6[5] = - 2 * (yd6(tk) - yde_0) * (c12345 + c123456)
    xi6[6] = - 2 * (yd6(tk) - yde_0) * (c123456)
    xi6[7] = 0
    xi6[8] = 0
    #joint 5
    dist5_m[tk] = (yd5(tk) - yde_0) ** 2
    xi5 = np.zeros((8,1))
    xi5[1] = - 2 * (yd5(tk) - yde_0) * (- s1 + c12 - s123 + c1234 + c12345)
    xi5[2] = - 2 * (yd5(tk) - yde_0) * (c12 - s123 + c1234 + c12345)
    xi5[3] = - 2 * (yd5(tk) - yde_0) * (- s123 + c1234 + c12345)
    xi5[4] = - 2 * (yd5(tk) - yde_0) * (c1234 + c12345)
    xi5[5] = - 2 * (yd5(tk) - yde_0) * (c12345)
    xi5[6] = 0
    xi5[7] = 0
    xi5[8] = 0
    k5 = 50
    k6 = 50
    k7 = 100
    task2[:,tk] = (np.eye(8) - Jac1_psinv * Jac1) * (k5 * xi5 + k6 * xi6 + k7 * xi7)
    #angular velocity
#qd_(tk, :) = task1' + task2(:, tk)';
    #****************** Parallagh A *******************
#     if max([dist5_m(tk), dist6_m(tk), dist7_m(tk)]) == dist5_m(tk)
#         task2(:,tk) = (eye(8) - Jac1_psinv*Jac1)*(20*xi5);
#     elseif max([dist5_m(tk), dist6_m(tk), dist7_m(tk)]) == dist6_m(tk)
#         task2(:,tk) = (eye(8) - Jac1_psinv*Jac1)*(20*xi6);
#     else
#         task2(:,tk) = (eye(8) - Jac1_psinv*Jac1)*(20*xi7);
#     end
#     qd_(tk, :) = task1' + task2(:, tk)';
    #****************** Parallagh B *******************
    if np.amax(np.array([dist5_m(tk),dist6_m(tk),dist7_m(tk)])) >= 0.01:
        qd_[tk,:] = np.transpose(task2(:,tk))
    else:
        qd_[tk,:] = np.transpose(task1) + np.transpose(task2(:,tk))
    #end-effector velocity
    v8[:,tk] = Jac1 * np.transpose(qd_(tk,:))
    # numerical integration --> kinematic simulation
    qd[tk + 1,1] = qd(tk,1) + dt * qd_(tk,1)
    qd[tk + 1,2] = qd(tk,2) + dt * qd_(tk,2)
    qd[tk + 1,3] = qd(tk,3) + dt * qd_(tk,3)
    qd[tk + 1,4] = qd(tk,4) + dt * qd_(tk,4)
    qd[tk + 1,5] = qd(tk,5) + dt * qd_(tk,5)
    qd[tk + 1,6] = qd(tk,6) + dt * qd_(tk,6)
    qd[tk + 1,7] = qd(tk,7) + dt * qd_(tk,7)
    qd[tk + 1,8] = qd(tk,8) + dt * qd_(tk,8)
    #plot robot
    if (np.mod(tk,dtk) == 0 or tk == 1):
        cla
        plt.plot(xd,yd,'m:')
        plt.plot(0,0,'o')
        plt.plot(np.array([0,xd1(tk)]),np.array([0,yd1(tk)]),'b')
        plt.plot(xd1(tk),yd1(tk),'ro')
        plt.plot(np.array([xd1(tk),xd2(tk)]),np.array([yd1(tk),yd2(tk)]),'b')
        plt.plot(xd2(tk),yd2(tk),'ro')
        plt.plot(np.array([xd2(tk),xd3(tk)]),np.array([yd2(tk),yd3(tk)]),'b')
        plt.plot(xd3(tk),yd3(tk),'ro')
        plt.plot(np.array([xd3(tk),xd4(tk)]),np.array([yd3(tk),yd4(tk)]),'b')
        plt.plot(xd4(tk),yd4(tk),'ro')
        plt.plot(np.array([xd4(tk),xd5(tk)]),np.array([yd4(tk),yd5(tk)]),'b')
        plt.plot(xd5(tk),yd5(tk),'ro')
        plt.plot(np.array([xd5(tk),xd6(tk)]),np.array([yd5(tk),yd6(tk)]),'b')
        plt.plot(xd6(tk),yd6(tk),'ro')
        plt.plot(np.array([xd6(tk),xd7(tk)]),np.array([yd6(tk),yd7(tk)]),'b')
        plt.plot(xd7(tk),yd7(tk),'ro')
        plt.plot(np.array([xd7(tk),xd8(tk)]),np.array([yd7(tk),yd8(tk)]),'b')
        plt.plot(xd8(tk),yd8(tk),'ro')
        plt.plot(xd8(tk),yd8(tk),'g+')
        #OBSTACLES
        diam = 0.4
        d = 1
        rectangle('Position',np.array([xde_0 - diam / 2,yde_0 - d / 2 - diam,diam,diam]),'Curvature',np.array([1,1]),'FaceColor','blue')
        rectangle('Position',np.array([xde_0 - diam / 2,yde_0 + d / 2,diam,diam]),'Curvature',np.array([1,1]),'FaceColor','blue')
        #########
        pause(pauseTime)
    tk = tk + 1
    tt = tt + dt


# PLOTS

#end-effector position
xd8 = np.array([xd8(1),xd8])
yd8 = np.array([yd8(1),yd8])
figure
subplot(1,2,1)
plt.plot(k,xd8)
plt.title('x_e')
subplot(1,2,2)
plt.plot(k,yd8)
plt.title('y_e')
#end-effector velocity
v8_x = v8(1,:)
v8_y = v8(2,:)
figure
subplot(1,2,1)
plt.plot(v8_x)
plt.title('$\dot{x}_e$','interpreter','latex')
subplot(1,2,2)
plt.plot(v8_y)
plt.title('$\dot{y}_e$','interpreter','latex')
#q
figure
subplot(2,4,1)
plt.plot(k,qd(:,1))
plt.title('q_1')
subplot(2,4,2)
plt.plot(k,qd(:,2))
plt.title('q_2')
subplot(2,4,3)
plt.plot(k,qd(:,3))
plt.title('q_3')
subplot(2,4,4)
plt.plot(k,qd(:,4))
plt.title('q_4')
subplot(2,4,5)
plt.plot(k,qd(:,5))
plt.title('q_5')
subplot(2,4,6)
plt.plot(k,qd(:,6))
plt.title('q_6')
subplot(2,4,7)
plt.plot(k,qd(:,7))
plt.title('q_7')
subplot(2,4,8)
plt.plot(k,qd(:,8))
plt.title('q_8')
#q_dot
qd_ = np.array([[np.zeros((1,8))],[qd_]])
figure
subplot(2,4,1)
plt.plot(k,qd_(:,1))
plt.title('$\dot{q}_1$','interpreter','latex')
subplot(2,4,2)
plt.plot(k,qd_(:,2))
plt.title('$\dot{q}_2$','interpreter','latex')
subplot(2,4,3)
plt.plot(k,qd_(:,3))
plt.title('$\dot{q}_3$','interpreter','latex')
subplot(2,4,4)
plt.plot(k,qd_(:,4))
plt.title('$\dot{q}_4$','interpreter','latex')
subplot(2,4,5)
plt.plot(k,qd_(:,5))
plt.title('$\dot{q}_5$','interpreter','latex')
subplot(2,4,6)
plt.plot(k,qd_(:,6))
plt.title('$\dot{q}_6$','interpreter','latex')
subplot(2,4,7)
plt.plot(k,qd_(:,7))
plt.title('$\dot{q}_7$','interpreter','latex')
subplot(2,4,8)
plt.plot(k,qd_(:,8))
plt.title('$\dot{q}_8$','interpreter','latex')
#task2
# task2 = [ones(8,1) task2];

# figure;
# subplot(2,4,1); plot(k, task2(1,:));
# title('$\dot{q}_1$','interpreter','latex');
# subplot(2,4,2); plot(k, task2(2,:));
# title('$\dot{q}_2$','interpreter','latex');
# subplot(2,4,3); plot(k, task2(3,:));
# title('$\dot{q}_3$','interpreter','latex');
# subplot(2,4,4); plot(k, task2(4,:));
# title('$\dot{q}_4$','interpreter','latex');
# subplot(2,4,5); plot(k, task2(5,:));
# title('$\dot{q}_5$','interpreter','latex');
# subplot(2,4,6); plot(k, task2(6,:));
# title('$\dot{q}_6$','interpreter','latex');
# subplot(2,4,7); plot(k, task2(7,:));
# title('$\dot{q}_7$','interpreter','latex');
# subplot(2,4,8); plot(k, task2(8,:));
# title('$\dot{q}_8$','interpreter','latex');

#distance joints - obstacles
dist5_m = np.array([0,dist5_m])
dist6_m = np.array([0,dist6_m])
dist7_m = np.array([0,dist7_m])
figure
subplot(1,3,1)
plt.plot(k,dist5_m)
plt.title('$(y_5(q) - y_m)^2$','interpreter','latex')
subplot(1,3,2)
plt.plot(k,dist6_m)
plt.title('$(y_6(q) - y_m)^2$','interpreter','latex')
subplot(1,3,3)
plt.plot(k,dist7_m)
plt.title('$(y_7(q) - y_m)^2$','interpreter','latex')