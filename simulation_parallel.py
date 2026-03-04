import os
import anuga
import numpy as np
import argparse

# os.environ["OMP_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1"


def run_simulation(input_file, output_dir, value):
    """
    A placeholder for the actual simulation logic.
    """
    print(f"Running simulation with:")
    print(f"  Input file: {input_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Value: {value}")

    # Your simulation code would go here.
    # For example, using anuga and numpy.
    pass

def calculate_elevation(x, y, channel_num, channel_pos, channel_state, branch_width, main_road_width, gate_thickness, wall_height, **kwargs):
    # --- 参数映射 ---
    import numpy as np
    num_junctions = channel_num
    junction_positions = channel_pos  
    branch_active = channel_state    

    z = np.zeros_like(x)
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_range = x_max - x_min
    mid_y = (y_min + y_max) / 2
    
    # 主干道的上下边界
    main_bottom = mid_y - main_road_width / 2
    main_top = mid_y + main_road_width / 2

    for i in range(len(x)):
        # 1. 基础判断：是否在主轴内（主干道永远是通路）
        is_on_main_road = (main_bottom <= y[i] <= main_top)
        
        is_on_passable_branch = False
        is_blocked_by_wall = False
        
        for idx in range(num_junctions):
            pos_ratio = junction_positions[idx]
            is_active = branch_active[idx]
            branch_center_x = x_min + x_range * pos_ratio
            
            # 判断坐标是否在当前支路的宽度范围内
            in_branch_x_range = (branch_center_x - branch_width/2 <= x[i] <= branch_center_x + branch_width/2)
            # 支路在主干道下方 (y轴较小的一侧)
            in_branch_y_range = (y[i] < main_bottom)
            
            if in_branch_x_range and in_branch_y_range:
                if is_active:
                    # 开启状态：支路全线通路
                    is_on_passable_branch = True
                else:
                    # 关闭状态：执行“两头封堵”逻辑
                    # A. 检查路口门 (靠近主干道边缘)
                    is_at_gate = (y[i] >= main_bottom - gate_thickness)
                    # B. 检查支路底部 (靠近 y_min 边缘)
                    is_at_bottom_wall = (y[i] <= y_min + gate_thickness)
                    
                    if is_at_gate or is_at_bottom_wall:
                        is_blocked_by_wall = True
                    else:
                        # 门和底墙之间的区域仍然是平地 (路面)
                        is_on_passable_branch = True

        # 2. 最终高度判定
        # 通路条件：在主干道上 OR 在开启/未被封堵的支路段上
        # 且 必须没有被特定的“门”或“底墙”挡住
        if (is_on_main_road or is_on_passable_branch) and not is_blocked_by_wall:
            z[i] = 0
        else:
            z[i] = wall_height
            
    return z









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a parallel simulation.')
    parser.add_argument('--x_max', type=float, required=True,
                        help=' ')
    parser.add_argument('--y_max', type=float, required=True,
                        help=' ')
    parser.add_argument('--sim_name', type=str, required=True,
                        help=' ')
    parser.add_argument('--channel_num', type=int, required=True,
                        help=' ')
    parser.add_argument('--channel_pos', type=float, nargs='+', required=True,
                        help=' ')
    parser.add_argument('--channel_state', type=int, nargs='+', required=True,
                        help=' ')
    parser.add_argument('--branch_width', type=float, required=True,
                        help=' ')
    parser.add_argument('--main_road_width', type=float, required=True,
                        help=' ')
    parser.add_argument('--gate_thickness', type=float, required=True,
                        help=' ')
    parser.add_argument('--wall_height', type=float, required=True,
                        help=' ')


    args = parser.parse_args()

    
    
    
    
    if anuga.myid == 0:

        # Set the initial conditions
        domain = anuga.rectangular_cross_domain(400, 40, len1=args.x_max, len2=args.y_max)
        domain.set_name(args.sim_name)


        params = {
            'channel_num': args.channel_num,
            'channel_pos': args.channel_pos,
            'channel_state': args.channel_state,
            'branch_width': args.branch_width,
            'main_road_width': args.main_road_width,
            'gate_thickness': args.gate_thickness,
            'wall_height': args.wall_height
        }
        
        topography = lambda x, y: calculate_elevation(x, y, **params)
        
        domain.set_quantity('elevation', topography, location='centroids')         # Use function for elevation
        domain.set_quantity('friction', 0.01, location='centroids')                # Constant friction
        # domain.set_quantity('stage', expression='elevation', location='centroids') # Dry Bed
        domain.set_quantity('stage', 0.3) # Dry Bed


    else:
        domain = None

    # Distribute the domain to all processes
    domain = anuga.distribute(domain)


    ##-----------------------------------------------------------------------
    ## Setup boundary conditions
    ##-----------------------------------------------------------------------
    
    Bi = anuga.Dirichlet_boundary([0.8, 0, 0])         # Inflow
    Bo = anuga.Dirichlet_boundary([0.1, 0, 0])          # Outflow
    Br = anuga.Reflective_boundary(domain)            # Solid reflective wall
    Boo = anuga.Dirichlet_boundary([0.1, 0, 0])
    
    domain.set_boundary({'left': Bi, 'right': Bo, 'top': Br, 'bottom': Boo})
    
    ##-----------------------------------------------------------------------
    ## Evolve system through time
    ##-----------------------------------------------------------------------
    

    for t in domain.evolve(yieldstep=2, duration=50):
        pass
    
        # domain.save_depth_frame(vmin=0.0,vmax=1.0)

    # Merge sww files on process 0
    domain.sww_merge()

    # Finalize MPI
    anuga.finalize()
