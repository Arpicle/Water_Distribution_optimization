class Water_Distribution():
    def __init__(self, sim_name, **kwargs):
        self.channel_num = kwargs.get('channel_num', 0)
        self.channel_pos = kwargs.get('channel_pos', [])
        self.branch_width = kwargs.get('branch_width', 1.0)
        self.main_road_width = kwargs.get('main_road_width', 2.0)
        self.wall_height = kwargs.get('wall_height', 0.5)
        self.gate_thickness = kwargs.get('gate_thickness', 0.5)
        self.channel_state = kwargs.get('channel_state', [])
        self.x_max = kwargs.get('x_max', 200)
        self.y_max = kwargs.get('y_max', 100)
        self.sim_name = sim_name


    def step(self, action, state, step_num):
        # Convert 1/0 integers to boolean True/False
        self.channel_state = [True if x == 1 else False for x in action]
        add_water = self.step_sim()
        diff = (state - add_water)
        reward = -(diff/state) ** 2 * (1 + 0.1*step_num)

        next_state = diff
        if (next_state < 0).any():
            done = True
        else:
            done = False
        return reward, next_state, done

    
    def step_sim(self):
        if len(self.channel_pos) != self.channel_num or len(self.channel_state) != self.channel_num:
            print("BAD CONSTRUCTION!")
            return 0

        tag = self.simulation(self.sim_name, self.channel_num, self.channel_pos, self.branch_width, self.main_road_width, self.wall_height, self.gate_thickness, self.x_max, self.y_max, self.channel_state, debug=False)

        gate_pos = self.get_gate_locations(channel_num=self.channel_num, channel_pos=self.channel_pos, branch_width=self.branch_width, main_road_width=self.main_road_width, gate_thickness=self.gate_thickness, x_max=self.x_max, y_max=self.y_max)

        file_path = os.path.join(os.path.dirname(__file__), self.sim_name + ".sww")
        water_volums = self.get_water_volum(file_path, gate_pos)
        # print(water_volums)
        return water_volums








    def calculate_elevation(self, x, y, channel_num, channel_pos, channel_state, branch_width, main_road_width, gate_thickness, wall_height, **kwargs):
        # --- 内部参数设置 ---
        num_junctions = channel_num
        junction_positions = channel_pos  # 支路在主轴的位置比例
        branch_active = channel_state   # 支路状态：True为开启，False为关闭
        
        # --------------------

        z = np.zeros_like(x)
        
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        x_range = x_max - x_min
        mid_y = (y_min + y_max) / 2
        
        # 主干道的上下边界
        main_bottom = mid_y - main_road_width / 2
        main_top = mid_y + main_road_width / 2

        for i in range(len(x)):
            # 1. 基础判断：是否在主干道内
            is_on_main_road = (main_bottom <= y[i] <= main_top)
            
            is_on_open_branch = False
            is_on_closed_branch_but_blocked = False
            
            for idx in range(num_junctions):
                pos_ratio = junction_positions[idx]
                is_active = branch_active[idx]
                branch_center_x = x_min + x_range * pos_ratio
                
                # 判断坐标是否在当前支路的宽度范围内
                in_branch_x_range = (branch_center_x - branch_width/2 <= x[i] <= branch_center_x + branch_width/2)
                # 支路在主干道下方
                in_branch_y_range = (y[i] < main_bottom)
                
                if in_branch_x_range and in_branch_y_range:
                    if is_active:
                        # 如果支路是开的，标记为通路
                        is_on_open_branch = True
                    else:
                        # 如果支路是关的
                        # 检查是否处于“门口”位置（靠近主干道边缘的厚度区域）
                        # 门口范围：从主干道边缘向下延伸 gate_thickness 的距离
                        if y[i] >= (main_bottom - gate_thickness):
                            is_on_closed_branch_but_blocked = True
                        else:
                            # 门后的支路空间仍视为通路（或者你也可以设为全是墙，这里设为通路）
                            is_on_open_branch = True

            # 2. 最终高度判定逻辑
            # 如果在主轴上，或者是开启的支路，高度为0
            if is_on_main_road or is_on_open_branch:
                z[i] = 0
            else:
                # 其他地方（野外、封闭的路口、支路间的空隙）全是墙
                z[i] = wall_height
                
        return z


    def simulation(self, sim_name, channel_num, channel_pos, branch_width, main_road_width, wall_height, gate_thickness, x_max, y_max, channel_state, debug=False):
        
        script_path = os.path.abspath(__file__)
        file_path = os.path.join(os.path.dirname(script_path), sim_name + ".sww")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        domain = anuga.rectangular_cross_domain(400, 40, len1=x_max, len2=y_max)
        domain.set_name(sim_name)

        if debug == True:
            rc('animation', html='jshtml')
            plt.figure(figsize=(10, 10))
            domain.triplot()
            ax = plt.gca()
            ax.set_aspect(1)
            plt.show()
            
        #----------------------------------------------------------------------
        # Setup initial conditions
        #----------------------------------------------------------------------

        # 使用 lambda 表达式将参数注入到函数中，同时保持接口为 (x, y)
        # 构造参数字典
        params = {
            'channel_num': channel_num,
            'channel_pos': channel_pos,
            'channel_state': channel_state,
            'branch_width': branch_width,
            'main_road_width': main_road_width,
            'gate_thickness': gate_thickness,
            'wall_height': wall_height
        }
        
        topography = lambda x, y: calculate_elevation(x, y, **params)
        
        domain.set_quantity('elevation', topography, location='centroids')         # Use function for elevation
        domain.set_quantity('friction', 0.01, location='centroids')                # Constant friction
        # domain.set_quantity('stage', expression='elevation', location='centroids') # Dry Bed
        domain.set_quantity('stage', 0.5) # Dry Bed

        if debug == True:
            plt.figure(figsize=(10, 10))
            domain.tripcolor(
                facecolors = domain.elev,
                edgecolors='k',
                cmap='Greys_r')
            plt.colorbar()
            plt.show()
        
        ##-----------------------------------------------------------------------
        ## Setup boundary conditions
        ##-----------------------------------------------------------------------
        
        Bi = anuga.Dirichlet_boundary([1.3, 0, 0])         # Inflow
        Bo = anuga.Dirichlet_boundary([-2, 0, 0])          # Outflow
        Br = anuga.Reflective_boundary(domain)            # Solid reflective wall
        
        domain.set_boundary({'left': Bi, 'right': Br, 'top': Br, 'bottom': Br})
        
        ##-----------------------------------------------------------------------
        ## Evolve system through time
        ##-----------------------------------------------------------------------
        

        for t in domain.evolve(yieldstep=2, duration=100):
        
            domain.save_depth_frame(vmin=0.0,vmax=1.0)
            if debug == True:
                domain.print_timestepping_statistics()
        
        # Read in the png files stored during the evolve loop
        # if debug == True:
        #     # plt.figure(figsize=(10, 10))
        #     domain.make_depth_animation()
        #     # plt.show()
        if debug:
            return domain
        else:
            return 0



    def get_water_volum(self, sww_file, gate_poss):
        from anuga.shallow_water.sww_interrogate import get_flow_through_cross_section
        water_volums = []
        for pos in gate_poss:
            time, flux = get_flow_through_cross_section(sww_file, pos)
            volum = np.trapz(flux, time)
            water_volums.append(volum)
        return water_volums



    def get_gate_locations(self, **kwargs):
        """
        计算每个丁字路口“门”的物理位置。
        返回格式：[[(x1, y1), (x2, y2)], ...]，其中每组坐标代表门的中轴线端点。
        """
        # 提取参数，使用默认值
        channel_num = kwargs.get('channel_num', 0)
        channel_pos = kwargs.get('channel_pos', [])
        branch_width = kwargs.get('branch_width', 1.0)
        main_road_width = kwargs.get('main_road_width', 2.0)
        gate_thickness = kwargs.get('gate_thickness', 0.5)
        
        # 定义空间范围
        x_min = kwargs.get('x_min', 0)
        x_max = kwargs.get('x_max', 200)
        y_min = kwargs.get('y_min', 0)
        y_max = kwargs.get('y_max', 100)
        
        x_range = x_max - x_min
        mid_y = (y_min + y_max) / 2
        
        # 门的中轴线 Y 坐标（位于主干道下方边缘，并考虑厚度的一半）
        # 主干道底边是 mid_y - main_road_width/2
        # 门位于从 main_bottom 向下延伸 gate_thickness 的区域
        # 所以门的中轴线 y 坐标大约是 main_bottom - gate_thickness / 2
        main_bottom = mid_y - main_road_width / 2
        gate_y_center = main_bottom - (gate_thickness / 2)
        
        gate_points = []
        for i in range(channel_num):
            # 计算当前支路的中心 X 坐标
            pos_ratio = channel_pos[i]
            branch_x_center = x_min + x_range * pos_ratio
            
            # 门所在的左右端点 X 坐标
            x_start = branch_x_center - branch_width / 2
            x_end = branch_x_center + branch_width / 2
            
            # 用两个点描述门的位置
            gate_points.append([(x_start, gate_y_center), (x_end, gate_y_center)])
            
        return gate_points