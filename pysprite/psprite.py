import numpy as np
import math


def deviation(data, u):
    return (sum([(i - u) ** 2 for i in data]) / (len(data) - 1)) ** .5


def deviation_dict(data, u):
    return (sum([((i - u) ** 2) * data[i] for i in data]) / (sum(data.values()) - 1)) ** .5


def SPRITE(u, mean_decimals, sd, sd_decimals, n, min_value, max_value, restrictions=[], random_start="Yes",
           min_start="No"):
    scale = range(min_value, max_value + 1)
    if u > max_value or u < min_value:
        return "Your mean is outside the scale"
    ##GRIM test and possible totals
    if round(round(u * n, 0) / n, mean_decimals) != u:
        return "grim test failed"
    lower = u - .5 / (10 ** mean_decimals)
    upper = u + .5 / (10 ** mean_decimals)
    l_bound = int(math.ceil(lower * n))
    if lower < 0:
        l_bound = int(lower * n)
    u_bound = int(upper * n)
    if upper < 0:
        u_bound = int(math.floor(upper * n))
    if restrictions:
        for i in scale:
            if i not in restrictions:
                break
        start = np.array((n - len(restrictions)) * [i])
        random_sum = np.random.choice(range(l_bound, u_bound + 1))
        loop_count = 0
        if sum(start) + sum(restrictions) == random_sum:
            random = start
        elif sum(start) + sum(restrictions) > random_sum:
            return "Your restrictions are impossible given the mean"
        else:
            escape = False
            while True:
                if escape:
                    break
                step = np.random.permutation([0] * (n - 1 - len(restrictions)) + [1])
                while True:
                    loop_count += 1
                    temp = start + step
                    if loop_count > 10000:
                        return "Your restrictions may be impossible"
                    if max(temp) > max(scale):
                        break
                    while True:
                        X = True
                        for i in restrictions:
                            if i in temp:
                                X = False
                        if X == True:
                            break
                        temp = temp + step
                    if max(temp) > max(scale):
                        break
                    if sum(temp) + sum(restrictions) > random_sum:
                        break

                    start = temp
                    if sum(start) + sum(restrictions) == random_sum:
                        escape = True
                        random = start
                        break
                    break
    else:
        ##create skew,flat, and random distributions

        ####################skew
        if random_start == "No":
            if l_bound == u_bound:
                local_u = l_bound / float(n)
                if max(scale) - local_u < local_u - min(scale):
                    skew = [max(scale)]
                else:
                    skew = [min(scale)]
                for i in range(n - 1):
                    if np.mean(skew) <= local_u:
                        skew.append(max(scale))
                    else:
                        skew.append(min(scale))

                skew.sort()
                if sum(skew) == l_bound:
                    pass
                else:
                    diff = l_bound - sum(skew)
                    if diff < 0:
                        skew[-1] = skew[-1] + diff
                    else:
                        skew[0] = skew[0] + diff
            else:
                max_sd = 0
                max_skew = []
                for i in range(l_bound, u_bound + 1):
                    local_u = i / float(n)
                    if max(scale) - local_u < local_u - min(scale):
                        temp_skew = [max(scale)]
                    else:
                        temp_skew = [min(scale)]
                    for ii in range(n - 1):
                        if np.mean(temp_skew) <= local_u:
                            temp_skew.append(max(scale))
                        else:
                            temp_skew.append(min(scale))

                    temp_skew.sort()
                    if sum(temp_skew) == i:
                        if deviation(temp_skew, local_u) > max_sd:
                            max_sd = deviation(temp_skew, local_u)
                            max_skew = temp_skew
                    else:
                        diff = i - sum(temp_skew)
                        if diff < 0:
                            temp_skew[-1] = temp_skew[-1] + diff
                        else:
                            temp_skew[0] = temp_skew[0] + diff
                        if deviation(temp_skew, local_u) > max_sd:
                            max_sd = deviation(temp_skew, local_u)
                            max_skew = temp_skew
                    skew = max_skew
            #################################flat
            if l_bound == u_bound:
                local_u = l_bound / float(n)
                flat = n * [int(local_u)]
                if l_bound > 0:
                    while sum(flat) < l_bound:
                        flat.sort()
                        flat[0] = flat[0] + 1
                else:
                    while sum(flat) > l_bound:
                        flat.sort(reverse=True)
                        flat[0] = flat[0] - 1
            else:
                min_sd = 1000
                min_skew = []
                for i in range(l_bound, u_bound + 1):
                    local_u = i / float(n)
                    temp_flat = n * [int(local_u)]
                    if l_bound > 0:
                        while sum(temp_flat) < i:
                            temp_flat.sort()
                            temp_flat[0] = temp_flat[0] + 1
                    else:
                        while sum(temp_flat) > i:
                            temp_flat.sort(reverse=True)
                            temp_flat[0] = temp_flat[0] - 1
                    if deviation(temp_flat, local_u) < min_sd:
                        min_sd = deviation(temp_flat, local_u)
                        min_skew = temp_flat
                flat = min_skew
        #####################random
        random_sum = np.random.choice(range(l_bound, u_bound + 1))
        random = np.array(n * [min(scale)])
        if sum(random) == random_sum:
            pass
        else:
            while True:
                temp_random = random + np.random.permutation([0] * (n - 1) + [1])
                if max(temp_random) > max(scale):
                    continue
                random = temp_random
                if sum(random) == random_sum:
                    break
    if not restrictions:
        if random_start == 'No':
            if min_start == "Yes":
                initial = flat
                closest_sd = deviation(random, np.mean(flat))
                closest = flat
            else:
                differences = [abs(deviation(flat, np.mean(skew)) - sd), abs(deviation(random, np.mean(random)) - sd),
                               abs(deviation(skew, np.mean(skew)) - sd)]
                closest = [flat, random, skew][differences.index(min(differences))]
                closest_sd = deviation(closest, np.mean(closest))
                initial = closest
        else:
            initial = random
            closest_sd = deviation(random, np.mean(random))
            closest = random
    else:
        initial = random
    data = {}
    for i in range(min(scale), max(scale) + 1):
        data[i] = 0
    for i in initial:
        data[i] = data.get(i) + 1
    for i in restrictions:
        data[i] = data.get(i, 0) + 1
    count = 0
    true_u = sum([i * data[i] for i in data]) / float(sum(data.values()))
    data_sd = deviation_dict(data, true_u)
    if restrictions:
        closest_sd = data_sd
        closest = data
    if round(data_sd, sd_decimals) == sd:
        return ['solution', data]

    ##random walk
    while True:
        count += 1
        if count > 50000:
            return ["no solution", closest, closest_sd]
        if data_sd > sd:
            if np.random.random() > .5:
                for first in np.random.permutation(scale[:-2]):
                    if data[first] != 0:
                        break
                if data[first] == 0:
                    return "first selection error"
                for second in np.random.permutation(scale[scale.index(first) + 2:]):
                    if data[second] != 0:
                        break
                if data[second] == 0:
                    continue
                while True:
                    if first + 1 not in restrictions and second - 1 not in restrictions \
                            and first not in restrictions and second not in restrictions \
                            and data[first] > 0 and data[second] > 0:
                        data[first] = data[first] - 1
                        data[first + 1] = data[first + 1] + 1
                        data[second] = data[second] - 1
                        data[second - 1] = data[second - 1] + 1
                        break

                    else:
                        first = first - 1
                        second = second + 1
                        if data.get(first) >= 0 and data.get(second) >= 0:
                            continue
                        else:
                            break

            else:
                for first in np.random.permutation(scale[2:]):
                    if data[first] != 0:
                        break
                if data[first] == 0:
                    return "first selection error"
                for second in np.random.permutation(scale[:scale.index(first) - 1]):
                    if data[second] != 0:
                        break
                if data[second] == 0:
                    continue
                while True:
                    if first - 1 not in restrictions and second + 1 not in restrictions \
                            and first not in restrictions and second not in restrictions \
                            and data[first] > 0 and data[second] > 0:
                        data[first] = data[first] - 1
                        data[first - 1] = data[first - 1] + 1
                        data[second] = data[second] - 1
                        data[second + 1] = data[second + 1] + 1
                        break
                    else:
                        first = first + 1
                        second = second - 1
                        if data.get(first) >= 0 and data.get(second) >= 0:
                            continue
                        else:
                            break
        else:
            for first in np.random.permutation(scale[1:-1]):
                if data[first] != 0:
                    break
            if data[first] == 0:
                return "first selection error"
            for second in np.random.permutation(scale[1:-1]):
                if data[second] != 0:
                    if first == second:
                        if data[first] > 1:
                            break
                        else:
                            continue
                    else:
                        break
            if first == second:
                if data[first] > 1:
                    pass
                else:
                    continue
            if data[second] == 0:
                continue
            if first >= second:
                while True:
                    if first + 1 not in restrictions and second - 1 not in restrictions \
                            and first not in restrictions and second not in restrictions \
                            and data[first] > 0 and data[second] > 0:
                        data[first] = data[first] - 1
                        data[first + 1] = data[first + 1] + 1
                        data[second] = data[second] - 1
                        data[second - 1] = data[second - 1] + 1
                        break
                    else:
                        first = first + 1
                        second = second - 1
                        if data.get(first) >= 0 and data.get(second) >= 0 and data.get(first + 1) and data.has_key(
                                second - 1):
                            continue
                        else:
                            break

            else:
                while True:
                    if first - 1 not in restrictions and second + 1 not in restrictions \
                            and first not in restrictions and second not in restrictions \
                            and data[first] > 0 and data[second] > 0:
                        data[first] = data[first] - 1
                        data[first - 1] = data[first - 1] + 1
                        data[second] = data[second] - 1
                        data[second + 1] = data[second + 1] + 1
                        break
                    else:
                        first = first - 1
                        second = second + 1
                        if data.get(first) >= 0 and data.get(second) >= 0 and data.has_key(first - 1) and data.has_key(
                                second + 1):
                            continue
                        else:
                            break

        data_sd = deviation_dict(data, true_u)
        if abs(sd - data_sd) < abs(sd - closest_sd):
            closest = data
            closest_sd = data_sd
        if round(data_sd, sd_decimals) == sd:
            return ['solution', data]