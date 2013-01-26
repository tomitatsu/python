#coding:shift-jis

import numpy as np
from scipy.linalg import norm
import cvxopt
import cvxopt.solvers
from pylab import *

# 全データの総数。
N = 10

# クラス1のデータ。
cls1 = [(1,6),
	    (2,10),
	    (3,8),
	    (4,11),
	    (5,9)]

# クラス2のデータ。
cls2 = [(1,3),
	    (2,2),
	    (3,4),
	    (4,1),
	    (5,3)]

# クラス1には1、クラス2には-1を割り振ってarrayにする。
t = []
for i in range(N/2):
	t += [1.0]
for i in range(N/2):
	t += [-1.0]
t = array(t)

"""データ行列Xをvsstackで作成。以下のようにスタック。X[1,0]で2とか。X[行,列]。
[[ 1  6]
 [ 2 10]
・・・
 [ 4  1]
 [ 5  3]]
"""
X = vstack((cls1, cls2))




# 線形カーネル。x*yの総和。
def kernel(x, y):
	return np.dot(x, y)

#10*10の0行列を用意する。
K = np.zeros((N, N))

# Kを作る。
for i in range(N):
	for j in range(N):
		K[i,j] += t[i]*t[j]*kernel(X[i],X[j])

#Kはこのmatrixにしないとらめ。
Q = cvxopt.matrix(K)

# マイナスにするのでこのpが必要。-1がN個の列ベクトル。
p = cvxopt.matrix(-np.ones(N))

# 不等式制約。
G = cvxopt.matrix(np.diag([-1.0]*N))       # 対角成分が-1のNxN行列
h = cvxopt.matrix(np.zeros(N))             # 0がN個の列ベクトル

# 等式制約。w*xの総和が0。
A = cvxopt.matrix(t, (1,N))                # N個の教師信号が要素の行ベクトル（1xN）
b = cvxopt.matrix(0.0)                     # 定数0.0

sol = cvxopt.solvers.qp(Q, p, G, h, A, b)  # 二次計画法でラグランジュ乗数aを求める
a = array(sol['x']).reshape(N)             # 'x'がaに対応する。ラグランジュ乗数aのリスト。




# サポートベクトルのインデックスのリストを。
S = []
for i in range(len(a)):
	if a[i] < 0.00001: # ほぼ0ならサポートベクトルじゃない。
		continue
	else:
		S += [i]

# wを計算。
w = np.zeros(2) #行列を用意。
for n in S:
	w += a[n]*t[n]*X[n]

# bを計算。
sum = 0
for n in S: # サポートベクトルの数だけforループ。
	temp = 0
	for m in S: # サポートベクトルの数だけforループ。
		temp += a[m]*t[m]*kernel(X[n],X[m])
	sum += (t[n]-temp)
b = sum/len(S)




# データをプロット。
for p in (cls1+cls2):
	plot(p[0],p[1],'bo')

# サポートベクトルを描画
for n in S:
	plot(X[n,0], X[n,1], 'ro')    

# 識別関数を描画。
def f(x1, w, b):
    return - (w[0] / w[1]) * x1 - (b / w[1])

# 識別境界を描画
x1 = np.linspace(-6, 6, 1000)
x2 = [f(x, w, b) for x in x1]
plot(x1, x2, 'g-')
    
xlim(0, 6)
ylim(0, 12)
show()

