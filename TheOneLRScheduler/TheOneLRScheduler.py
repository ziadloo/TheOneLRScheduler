import torch


class TheOneLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, points):
        if not ("x" in points[0] and "y" in points[0]):
            raise Exception("Invalid input. The starting point should be a coordinate.")
        if not ("x" in points[-1] and "y" in points[-1]):
            raise Exception("Invalid input. The ending point should be a coordinate.")

        curves = []
        for i in range(1, len(points)):
            if not (("control1" in points[i] and "control2" in points[i]) !=
                ("x" in points[i] and "y" in points[i])):
                raise Exception("Invalid input. The provided points could not be parsed as expected.")
            elif "control1" in points[i] and "control2" in points[i]:
                if "control1" in points[i-1] and "control2" in points[i-1]:
                    raise Exception("Invalid input. There cannot be two controller blocks in a row.")
                else:
                    continue
            elif "control1" in points[i-1] and "control2" in points[i-1]:
                curves.append({
                        "type": "bezier",
                        "from": {
                            "x": float(points[i-2]["x"]),
                            "y": float(points[i-2]["y"]),
                        },
                        "control1": {
                            "x": float(points[i-1]["control1"]["x"]),
                            "y": float(points[i-1]["control1"]["y"]),
                        },
                        "control2": {
                            "x": float(points[i-1]["control2"]["x"]),
                            "y": float(points[i-1]["control2"]["y"]),
                        },
                        "to": {
                            "x": float(points[i]["x"]),
                            "y": float(points[i]["y"]),
                        },
                    })
            elif points[i-1]["x"] != points[i]["x"]:
                curves.append({
                        "type": "line",
                        "from": {
                            "x": float(points[i-1]["x"]),
                            "y": float(points[i-1]["y"]),
                        },
                        "to": {
                            "x": float(points[i]["x"]),
                            "y": float(points[i]["y"]),
                        },
                    })

        scaling_factor = float(points[0]["y"])
        def _get_sample(i):
            def _bezier_curve(c, x, tolerance=1e-6):
                p0 = (c["from"]["x"], c["from"]["y"])
                p1 = (c["control1"]["x"], c["control1"]["y"])
                p2 = (c["control2"]["x"], c["control2"]["y"])
                p3 = (c["to"]["x"], c["to"]["y"])

                t_low = 0.0
                t_high = 1.0
                
                while t_high - t_low > tolerance:
                    t_mid = (t_low + t_high) / 2
                    
                    x_mid = ((1 - t_mid) ** 3) * p0[0] + 3 * ((1 - t_mid) ** 2) * t_mid * p1[0] + 3 * (1 - t_mid) * (t_mid ** 2) * p2[0] + (t_mid ** 3) * p3[0]
                    
                    if x_mid < x:
                        t_low = t_mid
                    else:
                        t_high = t_mid
                
                t = (t_low + t_high) / 2
                
                y = ((1 - t) ** 3) * p0[1] + 3 * ((1 - t) ** 2) * t * p1[1] + 3 * (1 - t) * (t ** 2) * p2[1] + (t ** 3) * p3[1]
                
                return y

            def _line_curve(c, x):
                start = (c["from"]["x"], c["from"]["y"])
                end = (c["to"]["x"], c["to"]["y"])
                X, Y = zip(start, end)
                slope = (end[1] - start[1]) / (end[0] - start[0])
                return start[1] + slope * (x-start[0])

            if i <= 0:
                return curves[0]["from"]["y"] / scaling_factor
            elif i >= curves[-1]["to"]["x"]:
                return curves[-1]["to"]["y"] / scaling_factor

            left = 0
            right = len(curves)-1
            while left < right:
                mid = left + (right - left) // 2
                if curves[mid]["to"]["x"] < i:
                    left = mid + 1
                else:
                    right = mid

            if curves[left]["type"] == "line":
                return _line_curve(curves[left], i) / scaling_factor
            else:
                return _bezier_curve(curves[left], i) / scaling_factor

        super().__init__(optimizer, _get_sample)
