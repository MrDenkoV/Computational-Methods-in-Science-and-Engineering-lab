import numpy as np
from matplotlib import pyplot as plt
import time


# Zadanie 1

print("\n\tZadanie 1\n")


def sumfloats(T):
    res = np.float32(0)
    for el in T:
        res += el
    return res


def errors(T, ex):
    wyn = ex, sumfloats(T)
    er = abs(wyn[0]-wyn[1]), abs(wyn[0]-wyn[1])/abs(wyn[1])
    print(f"Iteracyjny\nBłąd bezwzględny {er[0]}\nBłąd względny {er[1]}")
    return er


def lerr(N, v):
    suma = np.float32(0)
    res = []
    for i in range(1, N):
        suma += v
        if i % 25000 == 0:
            exp = v*i
            res.append(abs(exp - suma)/exp)
    return res


def sumbin(T):
    if len(T) <= 5:
        res = np.float32(0)
        for i in range(len(T)):
            res += T[i]
        return res
    return sumbin(T[len(T)//2:]) + sumbin(T[:len(T)//2])


def binerrors(T, ex):
    wyn = ex, sumbin(T)
    err = abs(wyn[0]-wyn[1]), abs(wyn[0]-wyn[1])/abs(wyn[1])
    print(f"Rekurencyjny\nBłąd bezwzględny - {err[0]}\nBłąd względny - {err[1]}")
    return err


N = 10**7
v = np.float32(0.53125)

# 1
print(f"suma {N}*{v} = {sumfloats(N*[v])}")

# 2
errors(N*[v], N*v)

# 3
plt.plot(lerr(N, v))

plt.show()

# 4 & 5
binerrors(N*[v], N*v)

# 6
stit = time.process_time()
sumfloats(N*[v])
enit = time.process_time()

stre = time.process_time()
sumbin(N*[v])
enre = time.process_time()

print(f"Czas iteracyjnego = {enit-stit}\nCzas rekurencyjnego = {enre-stre}")

# 7
sml = np.float32(v)
big = np.float32(v*N)
inp = [sml]*(N//2) + [big]*(N//2), sml*(N//2) + big*(N//2)
print(f"\n{N//2} * [{sml}] + {N//2} * [{big}]")
binerrors(*inp)


# Zadanie 2

print("\n\tZADANIE 2\n")


def kahana(T):
    suma = np.float32(0)
    err = np.float32(0)
    for i in T:
        y = i - err
        tmp = suma + y
        err = (tmp - suma) - y
        suma = tmp
    return suma


def kerrors(T, ex):
    wyn = ex, kahana(T)
    err = abs(wyn[0]-wyn[1]), abs(wyn[0]-wyn[1])/abs(wyn[1])
    print(f"Kahana\nBłąd bezwzględny - {err[0]}\nBłąd względny - {err[1]}\n")
    return err


# 1 & 2
print(f"\n{N} * [{v}]")
kerrors([v]*N, v*N)
print(f"\n{N//2} * [{sml}] + {N//2} * [{big}]")
kerrors(*inp)

# 3
stk = time.process_time()
kahana(N*[v])
enk = time.process_time()

stre = time.process_time()
sumbin(N*[v])
enre = time.process_time()

print(f"Czas Kahana = {enk-stk}\nCzas rekurencyjnego = {enre-stre}")


# Zadanie 3

print("\n\t Zadanie 3\n")


def dzeta(s, n, f=0, step=1):
    start = 1
    end = n
    if step != 1:
        start = n-1
        end = 0
    if f == 0:
        res = np.float32(0)
        for k in range(start, end, step):
            res += np.float32(1/(k**s))
    else:
        res = np.float64(0)
        s = np.float64(s)
        for k in range(start, end, step):
            res += np.float64(1/(k**s))
    return res


def eta(s, n, f=0, step=1):
    start = 1
    end = n
    if step != 1:
        start = n-1
        end = 0
    if f == 0:
        res = np.float32(0)
        for k in range(start, end, step):
            res += np.float32(((-1)**(k-1))*1/(k**s))
    else:
        res = np.float64(0)
        s = np.float64(s)
        for k in range(start, end, step):
            res += np.float64(((-1)**(k-1))*1/(k**s))
    return res


def solve(s, n):
    for i in s:
        for j in n:
            print(f"s = {i}\t n = {j}")
            print(f"Dzeta up - float32 = {dzeta(np.float32(i), j)}\t float64 = {dzeta(np.float64(i), j, 1)}")
            print(f"Dzeta down - float32 = {dzeta(np.float32(i), j, step=-1)}\t float64 = {dzeta(np.float64(i), j, 1, step=-1)}")
            print(f"Eta up - float32 = {eta(np.float32(i), j)}\t float64 = {eta(np.float64(i), j, 1)}")
            print(f"Eta down - float32 = {eta(np.float32(i), j, step=-1)}\t float64 = {eta(np.float64(i), j, 1, step=-1)}\n")


# 1
s = [np.float32(2), np.float32(3.6667), np.float32(5), np.float32(7.2), np.float32(10)]
n = [50, 100, 200, 500, 1000]

solve(s, n)


# Zadanie 4

print("\n\n\t Zadanie 4\n")


def logistic(x, r, n, skip=0, f=0):
    res = [x]
    if f==0:
        for i in range(n):
            x = np.float32(r*x*(1-x))
            if i >= skip:
                res += [x]
    else:
        for i in range(n):
            x = np.float64(r*x*(1-x))
            if i >= skip:
                res += [x]
    return res


def bifurka(x, n, rep, f=0):
    res = []
    rs = []
    if f==0:
        for i in range(1000, 4001, 1):
            res += logistic(x, np.float32(i/1000), n, n-rep, f)[1:]
            rs += rep * [np.float32(i/1000)]
    else:
        for i in range(1000, 4001, 1):
            res += logistic(x, np.float64(i/1000), n, n-rep, f)[1:]
            rs += rep * [np.float64(i/1000)]
    return rs, res


def draw(x):
    plt.clf()
    plt.figure(figsize=(10, 7))
    plt.plot(*bifurka(x, 101, 20), ls='', marker='.', markersize=1)
    plt.show()


def zer(r, x):
    i=0
    while x != 0:
        x = np.float32(r*x*(1-x))
        i+=1
    return i


# a)
x0 = np.float32(0.77)
r = np.float32(3.77)

draw(x0)
draw(np.float32(0.27))
draw(np.float32(0.5))

# b)
x0 = np.float32(0.77)
r = np.float32(3.77)
plt.clf()
plt.plot(logistic(x0, r, 100))
plt.show()

plt.clf()
plt.plot(logistic(np.float64(x0), np.float64(r), 100, f=1))
plt.show()

# c)
r = np.float32(4)

print(f"zeruje sie po {zer(r, x0)} iteracjach dla x0={x0}")
x0 = np.float32(0.23)
print(f"zeruje sie po {zer(r, x0)} iteracjach dla x0={x0}")
x0 = np.float32(0.5)
print(f"zeruje sie po {zer(r, x0)} iteracjach dla x0={x0}")
x0 = np.float32(0.13)
print(f"zeruje sie po {zer(r, x0)} iteracjach dla x0={x0}")
x0 = np.float32(0.9)
print(f"zeruje sie po {zer(r, x0)} iteracjach dla x0={x0}")
