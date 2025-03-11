from manim import *
import numpy as np

class BoundOptimization(Scene):
    def construct(self):
        # 1. 创建不显示数字和刻度的坐标轴
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-6, 5, 1],
            x_length=8,
            y_length=6,
            axis_config={"include_numbers": False, "include_ticks": False},
        ).to_edge(DOWN)
        self.play(Create(axes))
        
        # 2. 定义原始目标函数 l(θ)=-(θ-3)^2+4
        def l(x):
            return - (x - 3)**2 + 4

        # 3. 预设迭代点与对应的 A 参数
        theta_vals = [0, 1.2, 2.0, 2.5, 2.8, 3.0]
        A_values = [3.5, 3.0, 2.5, 2.0, 0.8]
        
        # 定义下界函数 g 的工厂函数：
        # g(x; θ^t) = -A*(x - θ^(t+1))² + [l(θ^t) + A*(θ^t-θ^(t+1))²]
        def g_factory(theta_t, theta_tp1, A):
            K = l(theta_t) + A * ((theta_t - theta_tp1)**2)
            return lambda x: -A*(x - theta_tp1)**2 + K
        
        # 定义函数：对当前迭代计算“补丁”
        # 求解 l(x)=g(x) 的交点 r1, r2，然后构造二次多项式 P(x)
        # 使得 P(r1)=l(r1), P(r2)=l(r2), P((r1+r2)/2)=g((r1+r2)/2)+δ
        def patch_for_iteration(g, theta_t, theta_tp1, A, delta=0.3):
            a_coeff = A - 1
            b_coeff = 6 - 2*A*theta_tp1
            c_coeff = -5 + A*(theta_tp1**2) - l(theta_t) - A*(theta_t - theta_tp1)**2
            roots = np.roots([a_coeff, b_coeff, c_coeff])
            real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-6])
            if len(real_roots) >= 2:
                r1, r2 = real_roots[0], real_roots[1]
            else:
                r1, r2 = theta_t, theta_tp1
            m = (r1 + r2) / 2
            M = np.array([
                [r1**2, r1, 1],
                [r2**2, r2, 1],
                [m**2, m, 1]
            ])
            rhs = np.array([l(r1), l(r2), g(m) + delta])
            a_p, b_p, c_p = np.linalg.solve(M, rhs)
            def patch_func(x):
                return a_p*x**2 + b_p*x + c_p
            return r1, r2, patch_func
        
        # 4. 预先对所有迭代区间打补丁，构造 l_correct
        patches = []
        for i in range(len(theta_vals)-1):
            x_t = theta_vals[i]
            x_tp1 = theta_vals[i+1]
            A = A_values[i]
            g = g_factory(x_t, x_tp1, A)
            r1, r2, patch_func = patch_for_iteration(g, x_t, x_tp1, A, delta=0.3)
            patches.append((r1, r2, patch_func))
        
        def l_correct(x):
            for (r1, r2, patch_func) in patches:
                if r1 <= x <= r2:
                    return patch_func(x)
            return l(x)
        
        # 5. 绘制补丁后的 l_correct（直接作为初始显示，不显示 base l）
        l_correct_graph = axes.plot(l_correct, x_range=[-1, 4], color=BLUE)
        l_correct_label = axes.get_graph_label(l_correct_graph, label="l(\\theta)")
        self.play(Create(l_correct_graph), Write(l_correct_label))
        self.wait(0.35)
        
        # 6. 进行每一步的 g 改进步骤，红点更新，同时保留 g 曲线和横轴的 θ 标签
        current_dot = None
        for i in range(len(theta_vals)-1):
            x_t = theta_vals[i]
            x_tp1 = theta_vals[i+1]
            A = A_values[i]
            g = g_factory(x_t, x_tp1, A)
            
            # 辅助虚线：从当前红点位置（在 l_correct 上）到横轴和纵轴
            start_point = axes.coords_to_point(x_t, l_correct(x_t))
            dashed_v = DashedLine(start_point, axes.coords_to_point(x_t, 0), color=YELLOW)
            dashed_h = DashedLine(start_point, axes.coords_to_point(0, l_correct(x_t)), color=YELLOW)
            self.play(Create(dashed_v), Create(dashed_h))
            self.wait(0.35)
            
            # 在横轴上标出当前迭代点 \(\theta^t\)（永久保留）
            theta_label = Tex(f"$\\theta^{{{i}}}$")
            theta_label.next_to(axes.coords_to_point(x_t, 0), DOWN)
            self.play(Write(theta_label))
            
            # 绘制当前的下界函数 g（曲线永久保留）
            margin = 0.35
            g_graph = axes.plot(g, x_range=[x_t - margin, x_tp1 + margin], color=GREEN)
            # 取 g 曲线左端点（x = x_t - margin）作为标签位置，并向下左偏移
            g_max_point = axes.coords_to_point(x_tp1, g(x_tp1))
            g_label = Tex(f"$g(\\theta;\\theta^{{{i}}})$")
            g_label.next_to(g_max_point, UP)
            # 显示当前 g 曲线和标签
            self.play(Create(g_graph), Write(g_label))
            self.wait(0.35)
            
            # 红点操作：如果不存在则创建，否则移动到当前起始位置
            if current_dot is None:
                current_dot = Dot(start_point, color=RED)
                self.play(FadeIn(current_dot))
            else:
                self.play(current_dot.animate.move_to(start_point))
            
            # 红点沿 g 曲线从 x_t 移动到 x_tp1
            def update_dot(mob, alpha):
                new_x = x_t + alpha * (x_tp1 - x_t)
                new_y = g(new_x)
                mob.move_to(axes.coords_to_point(new_x, new_y))
            self.play(UpdateFromAlphaFunc(current_dot, update_dot), run_time=2)
            
            # 红点到达 g 曲线最高点后，绘制垂直虚线连接 g 与 l_correct 的对应点
            max_point_g = axes.coords_to_point(x_tp1, g(x_tp1))
            corresponding_l_point = axes.coords_to_point(x_tp1, l_correct(x_tp1))
            vertical_line = DashedLine(max_point_g, corresponding_l_point, color=YELLOW)
            self.play(Create(vertical_line))
            self.wait(0.35)
            
            # 将红点移动到 l_correct 上的更新点
            self.play(current_dot.animate.move_to(corresponding_l_point), run_time=1)
            self.wait(0.35)
            
            # 清除辅助虚线和当前 g 的标签（g 曲线和横轴的 θ 标签保留）
            self.play(
                FadeOut(dashed_v),
                FadeOut(dashed_h),
                FadeOut(vertical_line),
                FadeOut(g_label)
            )
        
        # 最终在最优点标记 “Optimum”
        final_point = axes.coords_to_point(theta_vals[-1], l_correct(theta_vals[-1]))
        final_label = Tex("Optimum").next_to(final_point, UP)
        self.play(Write(final_label))
        self.wait(1)