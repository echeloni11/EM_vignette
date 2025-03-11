from manim import *
import numpy as np

class BoundOptimization(Scene):
    def construct(self):
        # 1. Create axes without numbers and ticks
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-6, 5, 1],
            x_length=8,
            y_length=6,
            axis_config={"include_numbers": False, "include_ticks": False},
        ).to_edge(DOWN)
        self.play(Create(axes))
        
        # 2. Define the original objective function l(θ)=-(θ-3)^2+4
        def l(x):
            return - (x - 3)**2 + 4

        # 3. Define the iteration points and corresponding A values
        theta_vals = [0, 1.2, 2.0, 2.5, 2.8, 3.0]
        A_values = [3.5, 3.0, 2.5, 2.0, 0.8]
        
        # 4. Define the factory function for the lower bound function g
        # g(x; θ^t) = -A*(x - θ^(t+1))² + [l(θ^t) + A*(θ^t-θ^(t+1))²]
        def g_factory(theta_t, theta_tp1, A):
            K = l(theta_t) + A * ((theta_t - theta_tp1)**2)
            return lambda x: -A*(x - theta_tp1)**2 + K
        
        # Define a function to patch the lower bound function g
        # Solve l(x)=g(x) for the roots r1, r2, then construct a quadratic polynomial P(x)
        # such that P(r1)=l(r1), P(r2)=l(r2), P((r1+r2)/2)=g((r1+r2)/2)+δ
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
        
        # 5. Pre-patch all iteration intervals to construct l_correct
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
        
        # 6. Draw the patched l_correct (without the base l)
        l_correct_graph = axes.plot(l_correct, x_range=[-1, 4], color=BLUE)
        l_correct_label = axes.get_graph_label(l_correct_graph, label="l(\\theta)")
        self.play(Create(l_correct_graph), Write(l_correct_label))
        self.wait(0.35)
        
        # 7. Iterate through each step of the optimization
        current_dot = None
        for i in range(len(theta_vals)-1):
            x_t = theta_vals[i]
            x_tp1 = theta_vals[i+1]
            A = A_values[i]
            g = g_factory(x_t, x_tp1, A)
            
            # Auxiliary dashed lines: from the current red dot to the x-axis and y-axis
            start_point = axes.coords_to_point(x_t, l_correct(x_t))
            dashed_v = DashedLine(start_point, axes.coords_to_point(x_t, 0), color=YELLOW)
            dashed_h = DashedLine(start_point, axes.coords_to_point(0, l_correct(x_t)), color=YELLOW)
            self.play(Create(dashed_v), Create(dashed_h))
            self.wait(0.35)
            
            # Place the label for the current iteration point on the x-axis
            theta_label = Tex(f"$\\theta^{{{i}}}$")
            theta_label.next_to(axes.coords_to_point(x_t, 0), DOWN)
            self.play(Write(theta_label))
            
            # plot g with a small margin around the x_t and x_tp1
            margin = 0.35
            g_graph = axes.plot(g, x_range=[x_t - margin, x_tp1 + margin], color=GREEN)
            g_max_point = axes.coords_to_point(x_tp1, g(x_tp1))
            g_label = Tex(f"$g(\\theta;\\theta^{{{i}}})$")
            g_label.next_to(g_max_point, UP)
            self.play(Create(g_graph), Write(g_label))
            self.wait(0.35)
            
            if current_dot is None:
                current_dot = Dot(start_point, color=RED)
                self.play(FadeIn(current_dot))
            else:
                self.play(current_dot.animate.move_to(start_point))
            
            def update_dot(mob, alpha):
                new_x = x_t + alpha * (x_tp1 - x_t)
                new_y = g(new_x)
                mob.move_to(axes.coords_to_point(new_x, new_y))
            self.play(UpdateFromAlphaFunc(current_dot, update_dot), run_time=2)
            
            max_point_g = axes.coords_to_point(x_tp1, g(x_tp1))
            corresponding_l_point = axes.coords_to_point(x_tp1, l_correct(x_tp1))
            vertical_line = DashedLine(max_point_g, corresponding_l_point, color=YELLOW)
            self.play(Create(vertical_line))
            self.wait(0.35)
            
            self.play(current_dot.animate.move_to(corresponding_l_point), run_time=1)
            self.wait(0.35)
            
            self.play(
                FadeOut(dashed_v),
                FadeOut(dashed_h),
                FadeOut(vertical_line),
                FadeOut(g_label)
            )
        
        final_point = axes.coords_to_point(theta_vals[-1], l_correct(theta_vals[-1]))
        final_label = Tex("Optimum").next_to(final_point, UP)
        self.play(Write(final_label))
        self.wait(1)