import torch


class TheOneLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, points):
        curves = []
        for i in range(1, len(points)):
            if "control1" in points[i]:
                continue
            if "control1" in points[i-1]:
                curves.append(self._bezier_curve([(points[i-2]["x"], points[i-2]["y"]),
                                                  (points[i-1]["control1"]["x"], points[i-1]["control1"]["y"]),
                                                  (points[i-1]["control2"]["x"], points[i-1]["control2"]["y"]),
                                                  (points[i]["x"], points[i]["y"])]))
            else:
                p1 = (points[i-1]["x"], points[i-1]["y"])
                p2 = (points[i]["x"], points[i]["y"])
                print(p1, p2)
                if p1[0] != p2[0]:
                    curves.append(self._line_curve(p1, p2))
        samples = []
        for c in curves:
            if len(c) == 0:
                continue
            elif len(samples) == 0:
                samples = samples + [p[1] for p in c]
            elif c[0][0] == len(samples)-1:
                samples[-1] = min(samples[-1], c[0][1])
                samples = samples + [p[1] for p in c[1:]]
            else:
                samples = samples + [p[1] for p in c]

        samples = [s/samples[0] for s in samples]
        def _get_sample(i):
            if i < 0:
                return samples[0]
            if i < len(samples):
                return samples[i]
            else:
                return samples[-1]
        super().__init__(optimizer, _get_sample)


    def _binomial_coefficient(self, n, k):
        # Compute binomial coefficient C(n, k)
        result = 1
        for i in range(1, k + 1):
            result *= (n - i + 1) / i
        return result


    def _bezier_curve(self, coordinates):
        X, Y = zip(*coordinates)
        n = len(coordinates) - 1

        x_values = list(range(min(X), max(X) + 1))
        result = []

        for x in x_values:
            Y_val = 0
            for i, (xi, yi) in enumerate(coordinates):
                binomial_coeff = self._binomial_coefficient(n, i)
                Y_val += binomial_coeff * (1 - (x - min(X)) / (max(X) - min(X))) ** (n - i) * ((x - min(X)) / (max(X) - min(X))) ** i * yi

            result.append((x, Y_val))

        return result


    def _line_curve(self, start, end):
        X, Y = zip(start, end)
        slope = (end[1] - start[1]) / (end[0] - start[0])
        x_values = list(range(min(X), max(X) + 1))
        result = [(x, start[1] + slope * (x-start[0])) for x in x_values]
        return result
