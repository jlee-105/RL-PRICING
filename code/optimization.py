""" This is the code JJ made for SRP Project with Dr. Runger"""
from pyscipopt import Model, quicksum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def solve_wta_scip(I, T, D, R, U, H, P, Time_Limit):
    
    
    """
    Solves the SCHEDULING PROBLEM
    """
    M =200  # BIG M
    
    model = Model("SRP_SCHEDULE")

    # Decifne Decision variables
    s = {(i): model.addVar(vtype="INTEGER", lb=0, name=f"s_{i}")    # Start time of each task
         for i in range(I)}

    c = {(i): model.addVar(vtype="INTEGER", lb=0, name=f"c_{i}")    # End time of each task
         for i in range(I)}

    x = {(i, j): model.addVar(vtype="BINARY", name=f"x_{i}_{j}")    # Binary variable if i is precede for j then 1, o/w =0
         for i in range(I) for j in range(I)}

    y = {(i, t): model.addVar(vtype="BINARY", name=f"y_{i}_{t}")    # Binary variable if task i is in progress at time t then 1, o/w =0
          for i in range(I) for t in range(T)}
    
    #z = {model.addVar(vtype = "INTEGER", lb=0, name=f"objective")}
    z = model.addVar(vtype="INTEGER", lb=0, name="objective")

    # define objective function    
    obj_expr = z
        
    # 1) Objective function constraint
    for i in range(I):
        model.addCons(c[i] <=z)   
        
    # 2) Completion time should be same with start + duration
    for i in range(I):
        model.addCons(s[i] + D[i] == c[i])    

    # 3) Start time should be before the completion
    for i in range(I):
        model.addCons(s[i] <= c[i])  
        
    # 4) Precedence Constraint should be given, then start time should be after completion
    for i in range(I):
        for j in range(I):
        # Skip the diagonal case
            model.addCons(x[i, j] == P[i][j])
            if i != j:                 
                model.addCons(s[j] >= c[i] - M*(1 - x[i, j]))              

    #5) Relation b/w y and start time/completion time
    for i in range(I):
        for t in range(T):
              model.addCons(t >= s[i] - M*(1 - y[i,t]))
              model.addCons(t + 1 <= c[i] + M*(1 - y[i,t]))  
    
    #6) Resource should be less than total
    for r in range(R):
        for t in range(T):
            model.addCons(quicksum(y[i,t]*U[i][r] for i in range(I)) <= H[r])            
    
    #7) All progress should be like Duration        
    for i in range(I):
        model.addCons(quicksum(y[i,t] for t in range(T)) == D[i])         
   

    # model minimize  
    model.setObjective(obj_expr, "minimize")
   
    model.setParam("numerics/feastol", 1e-4)
    #model.setParam("numerics/lpfeastol", 1e-9)
    model.setParam("numerics/dualfeastol", 1e-4)
    model.setParam("limits/time", Time_Limit)

    # Solve model
    model.optimize()

    if model.getNSols() == 0:
        print("No feasible solution found!")
        return None, None, None

    best_sol = model.getBestSol()
    obj_value = model.getObjVal()
    
    task_schedule = np.zeros((I, 2), dtype = int)
    precedence = np.zeros((I, I), dtype = int)
    progress = np.zeros((I, T), dtype = int)
    #end_time = np.zeros((I), dtype = int)
    
    # start time :
    for i in range(I):
        task_schedule[i][0] = model.getSolVal(best_sol, s[i])
        task_schedule[i][1] = model.getSolVal(best_sol, c[i])
        
    for i in range(I):
        for j in range(I):    
            precedence[i][j] = model.getSolVal(best_sol, x[i, j])
            
            
    for i in range(I):
        for t in range(T):    
            progress[i][t] = model.getSolVal(best_sol, y[i, t])   
            
            
    usage_data = []
    for i in range(I):
        for r in range(R):
            if U[i][r] > 0:  # Only include resources actually used by job i
                start_time = task_schedule[i][0]
                end_time = task_schedule[i][1]
                usage_data.append({
                    "Job": i,
                    "Resource": r,
                    "Label": f"Job {i} - R{r}",
                    "Start": start_time,
                    "End": end_time,
                    "Duration": end_time - start_time,
                    "Usage": U[i][r]
                })

    df_usage = pd.DataFrame(usage_data)
    # Sort by resource or job (whatever you prefer)
    df_usage.sort_values(by=["Resource", "Job"], inplace=True)
    df_usage             
    
    
    print(task_schedule)    
    print(precedence) 
    print(progress)
    a = input()


    return task_schedule, best_sol, obj_value, usage_data


I = 4                    # NUMBER OF TASKS
T = 20                  # Time frame
R = 4                    # Types of resource
H = [3, 4, 5, 4]         # Number of resource
D = [3, 5, 7, 4]         # duration of the work 
U = [[2, 2, 1, 0], [1, 0, 1, 3], [3, 2, 0, 0], [0, 0, 1, 2]]
P = [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
Resource_Name = ['Electrician', 'Plummer', 'Welder', 'Glazier']


Time_limit = 30


task_schedule, best_sol, objective_value, usage_data = solve_wta_scip(I=I, T=T, R=R, D=D, U=U, H=H, P=P, Time_Limit=Time_limit)


# Convert to DataFrame
df_schedule = pd.DataFrame([
    {"Task": f"Task {i}", "Start": task_schedule[i][0], "End": task_schedule[i][1], "Duration": D[i]}
    for i in range(I)
]).sort_values(by="Start")

# ---------------------------------------------- #
# Step 2: Generate Gantt Chart for Tasks         #
# ---------------------------------------------- #

fig, ax = plt.subplots(figsize=(12, 6))
for idx, row in df_schedule.iterrows():
    ax.barh(row["Task"], row["Duration"], left=row["Start"], color='skyblue', edgecolor='black')

ax.set_xlabel("Time")
ax.set_ylabel("Tasks")
ax.set_title("Gantt Chart - Task Scheduling")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.show()



# ---------------------------------------------- #
# Step 3: Generate Gantt Chart for Resource Usage #
# ---------------------------------------------- #
# # Define time horizon based on max completion time
time_horizon = int(max(task_schedule[i][1] for i in range(I))) + 2  # Add margin
time_points = np.arange(time_horizon)

fig, ax = plt.subplots(figsize=(12, 6))

# Plot step function for each resource
for r in range(R):
    resource_usage = np.zeros(time_horizon)  # Extend to include time=0 correctly
    for t in range(time_horizon):
        resource_usage[t] = sum(U[i][r] for i in range(I) if task_schedule[i][0] <= t < task_schedule[i][1])
    
    baseline = r * (max(H) + 2)  # Baseline for each resource to create separation
    ax.step(time_points, resource_usage + baseline, where="pre", label=f"Resource {r}", linewidth=2)
    
    # Add annotations at the center of each step, including the initial value
    for t in range(time_horizon):
        if t == 0 or resource_usage[t] != resource_usage[t-1]:  # Show initial and changed values
            midpoint = t - 0.5 if t > 0 else t  # Adjust midpoint for initial value
            ax.text(midpoint, resource_usage[t] + baseline + 0.5, f"{int(resource_usage[t])}", 
                    fontsize=10, ha="center", va="bottom", color="black")

# Formatting
ax.set_xlabel("Time")
ax.set_ylabel("Resource Type")
ax.set_title("Resource Consumption Over Time")
plt.grid(axis="x", linestyle="--", alpha=0.5)

# Y-axis ticks for each resource's baseline
ax.set_yticks([(max(H) + 2) * r for r in range(R)])
ax.set_yticklabels(Resource_Name[r] for r in range(R))

plt.tight_layout()
plt.show()