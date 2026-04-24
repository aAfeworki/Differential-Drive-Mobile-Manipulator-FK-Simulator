import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Parameters
# ---------------------------
Lm1, Lm2, Lm3 = 0.1, 0.4, 0.3
L = 0.36
r = 0.07
Ts = 0.05

body_len = 0.6
body_wid = L + 0.2
body_h = 0.15

wheel_r = 0.07
wheel_w = 0.05

# ---------------------------
# State
# ---------------------------
X, Y, gamma = 0.0, 0.0, 0.0
wl, wr = 0.0, 0.0
V, w = 0.0, 0.0
t1 = t2 = t3 = 0.0

path_x, path_y = [X], [Y]
trace_on, show_icr = True, True
_running = {"flag": True}
sync_flag = {"lock": False}

# ---------------------------
# Kinematics
# ---------------------------
def wheels_to_Vw(wl, wr):
    Vl = r * wl
    Vr = r * wr
    return (Vr + Vl)/2, (Vr - Vl)/L

def Vw_to_wheels(V, w):
    Vr = V + 0.5 * L * w
    Vl = V - 0.5 * L * w
    return Vl/r, Vr/r

def icr(V, w):
    if abs(w) < 1e-6:
        return np.inf, np.nan, np.nan
    R = V/w
    return R, X - R*np.sin(gamma), Y + R*np.cos(gamma)

# ---------------------------
# Arm FK
# ---------------------------
def get_arm():
    p0 = np.array([0,0,0])
    p1 = np.array([0,0,Lm1])
    p2 = np.array([
        Lm2*np.cos(t2)*np.cos(t1),
        Lm2*np.cos(t2)*np.sin(t1),
        Lm1 + Lm2*np.sin(t2)
    ])
    p3 = np.array([
        (Lm2*np.cos(t2)+Lm3*np.cos(t2+t3))*np.cos(t1),
        (Lm2*np.cos(t2)+Lm3*np.cos(t2+t3))*np.sin(t1),
        Lm1 + Lm2*np.sin(t2) + Lm3*np.sin(t2+t3)
    ])
    return [p0,p1,p2,p3]

# ---------------------------
# Drawing
# ---------------------------
def rot2d(th):
    c,s=np.cos(th),np.sin(th)
    return np.array([[c,-s],[s,c]])

def draw_box():
    l,w_,h = body_len, body_wid, body_h
    x = np.array([ l/2, l/2,-l/2,-l/2, l/2])
    y = np.array([ w_/2,-w_/2,-w_/2, w_/2, w_/2])

    R = rot2d(gamma)
    pts = (R @ np.vstack((x,y))).T + [X,Y]

    # bottom + top
    ax.plot(pts[:,0],pts[:,1],0,'k')
    ax.plot(pts[:,0],pts[:,1],h,'k')

    # vertical edges 
    for i in range(4):
        ax.plot([pts[i,0],pts[i,0]],
                [pts[i,1],pts[i,1]],
                [0,h],'k')

def draw_wheel(offset_y):
    theta = np.linspace(0,2*np.pi,30)
    z = np.linspace(-wheel_w/2,wheel_w/2,10)
    theta,z = np.meshgrid(theta,z)

    # Cylinder in local frame (axis = y)
    x = wheel_r*np.cos(theta)
    y = z
    zc = wheel_r*np.sin(theta)

    # Flatten for rotation
    pts = np.vstack((x.flatten(), y.flatten()))

    # Rotate wheel orientation with robot heading
    R = rot2d(gamma)
    pts_rot = R @ pts

    x_rot = pts_rot[0].reshape(theta.shape)
    y_rot = pts_rot[1].reshape(theta.shape)

    # Rotate wheel center
    base = R @ np.array([0,offset_y]) + [X,Y]

    # Draw
    ax.plot_surface(x_rot + base[0],
                    y_rot + base[1],
                    zc + wheel_r,
                    alpha=0.9)

# ---------------------------
# Scene
# ---------------------------
def update_scene():
    ax.cla()
    ax.set_xlim(X-2,X+2)
    ax.set_ylim(Y-2,Y+2)
    ax.set_zlim(0,2)

    draw_box()
    draw_wheel(-L/2)
    draw_wheel( L/2)

    if trace_on:
        ax.plot(path_x,path_y,0,'g--')

    R_icr,x_icr,y_icr = icr(V,w)
    if show_icr and np.isfinite(R_icr):
        ax.scatter(x_icr,y_icr,0,c='r',marker='x')

    # Arm
    Rz = np.array([
        [np.cos(gamma),-np.sin(gamma),0],
        [np.sin(gamma), np.cos(gamma),0],
        [0,0,1]
    ])

    base = np.array([X,Y,body_h])
    joints = [(Rz @ p)+base for p in get_arm()]

    for i in range(3):
        ax.plot([joints[i][0],joints[i+1][0]],
                [joints[i][1],joints[i+1][1]],
                [joints[i][2],joints[i+1][2]],
                'r',lw=4,marker='o')

    ee=joints[-1]

    info = (
        f"V:{V:.2f}\n"
        f"w:{w:.2f}\n"
        f"ICR:{R_icr:.2f}\n"
        f"α:{gamma:.2f}\n"
        f"wl:{wl:.2f}, wr:{wr:.2f}\n"
        f"X:{X:.2f}, Y:{Y:.2f}\n"
        f"EE:[{ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f}]"
    )

    ax.text2D(0.95,0.95,info,transform=ax.transAxes,
              ha='right',va='top',
              bbox=dict(boxstyle="round",fc="white"))

# ---------------------------
# Simulation
# ---------------------------
def step():
    global X,Y,gamma,V,w

    if not _running["flag"]:
        return

    V,w = wheels_to_Vw(wl,wr)

    gamma += w*Ts
    X += V*Ts*np.cos(gamma)
    Y += V*Ts*np.sin(gamma)

    if trace_on:
        path_x.append(X)
        path_y.append(Y)

    update_scene()
    fig.canvas.draw_idle()

# ---------------------------
# UI
# ---------------------------
fig = plt.figure(figsize=(11,7))
ax = fig.add_axes([0.05,0.08,0.65,0.88],projection='3d')

axcolor="lightgoldenrodyellow"

s_wl=Slider(fig.add_axes([0.75,0.88,0.18,0.03],facecolor=axcolor),"wl",-10,10,valinit=0)
s_wr=Slider(fig.add_axes([0.75,0.84,0.18,0.03],facecolor=axcolor),"wr",-10,10,valinit=0)

s_V=Slider(fig.add_axes([0.75,0.78,0.18,0.03],facecolor=axcolor),"V",-2,2,valinit=0)
s_w=Slider(fig.add_axes([0.75,0.74,0.18,0.03],facecolor=axcolor),"w",-5,5,valinit=0)

s_t1=Slider(fig.add_axes([0.75,0.66,0.18,0.03],facecolor=axcolor),"θ1",-np.pi,np.pi,valinit=0)
s_t2=Slider(fig.add_axes([0.75,0.62,0.18,0.03],facecolor=axcolor),"θ2",-0.2,3.34,valinit=0)
s_t3=Slider(fig.add_axes([0.75,0.58,0.18,0.03],facecolor=axcolor),"θ3",-np.pi/2,np.pi/2,valinit=0)

# --- Synchronization ---
def update_from_wheels(val):
    global wl,wr,V,w
    if sync_flag["lock"]: return
    sync_flag["lock"]=True

    wl,wr = s_wl.val,s_wr.val
    V,w = wheels_to_Vw(wl,wr)

    s_V.set_val(V)
    s_w.set_val(w)

    sync_flag["lock"]=False

def update_from_body(val):
    global wl,wr,V,w
    if sync_flag["lock"]: return
    sync_flag["lock"]=True

    V,w = s_V.val,s_w.val
    wl,wr = Vw_to_wheels(V,w)

    s_wl.set_val(wl)
    s_wr.set_val(wr)

    sync_flag["lock"]=False

def update_arm(val):
    global t1,t2,t3
    t1,t2,t3 = s_t1.val,s_t2.val,s_t3.val

s_wl.on_changed(update_from_wheels)
s_wr.on_changed(update_from_wheels)
s_V.on_changed(update_from_body)
s_w.on_changed(update_from_body)
s_t1.on_changed(update_arm)
s_t2.on_changed(update_arm)
s_t3.on_changed(update_arm)

# Buttons
btn_run=Button(fig.add_axes([0.75,0.50,0.08,0.05]),"Pause")
btn_reset=Button(fig.add_axes([0.85,0.50,0.08,0.05]),"Reset")

def toggle(e):
    _running["flag"]=not _running["flag"]
    btn_run.label.set_text("Pause" if _running["flag"] else "Play")

def reset(e):
    global X,Y,gamma,path_x,path_y
    X,Y,gamma=0,0,0
    path_x,path_y=[X],[Y]

btn_run.on_clicked(toggle)
btn_reset.on_clicked(reset)

chk=CheckButtons(fig.add_axes([0.75,0.35,0.18,0.12]),
                 ["Trace Path","Show ICR"],[True,True])

def chk_fn(label):
    global trace_on,show_icr
    if label=="Trace Path": trace_on=not trace_on
    if label=="Show ICR": show_icr=not show_icr

chk.on_clicked(chk_fn)

# Timer
timer=fig.canvas.new_timer(interval=int(Ts*1000))
timer.add_callback(step)
timer.start()

update_scene()
plt.show()