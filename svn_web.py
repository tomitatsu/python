#coding:shift-jis

import numpy as np
from scipy.linalg import norm
import cvxopt
import cvxopt.solvers
from pylab import *

# �S�f�[�^�̑����B
N = 10

# �N���X1�̃f�[�^�B
cls1 = [(1,6),
	    (2,10),
	    (3,8),
	    (4,11),
	    (5,9)]

# �N���X2�̃f�[�^�B
cls2 = [(1,3),
	    (2,2),
	    (3,4),
	    (4,1),
	    (5,3)]

# �N���X1�ɂ�1�A�N���X2�ɂ�-1������U����array�ɂ���B
t = []
for i in range(N/2):
	t += [1.0]
for i in range(N/2):
	t += [-1.0]
t = array(t)

"""�f�[�^�s��X��vsstack�ō쐬�B�ȉ��̂悤�ɃX�^�b�N�BX[1,0]��2�Ƃ��BX[�s,��]�B
[[ 1  6]
 [ 2 10]
�E�E�E
 [ 4  1]
 [ 5  3]]
"""
X = vstack((cls1, cls2))




# ���`�J�[�l���Bx*y�̑��a�B
def kernel(x, y):
	return np.dot(x, y)

#10*10��0�s���p�ӂ���B
K = np.zeros((N, N))

# K�����B
for i in range(N):
	for j in range(N):
		K[i,j] += t[i]*t[j]*kernel(X[i],X[j])

#K�͂���matrix�ɂ��Ȃ��Ƃ�߁B
Q = cvxopt.matrix(K)

# �}�C�i�X�ɂ���̂ł���p���K�v�B-1��N�̗�x�N�g���B
p = cvxopt.matrix(-np.ones(N))

# �s��������B
G = cvxopt.matrix(np.diag([-1.0]*N))       # �Ίp������-1��NxN�s��
h = cvxopt.matrix(np.zeros(N))             # 0��N�̗�x�N�g��

# ��������Bw*x�̑��a��0�B
A = cvxopt.matrix(t, (1,N))                # N�̋��t�M�����v�f�̍s�x�N�g���i1xN�j
b = cvxopt.matrix(0.0)                     # �萔0.0

sol = cvxopt.solvers.qp(Q, p, G, h, A, b)  # �񎟌v��@�Ń��O�����W���搔a�����߂�
a = array(sol['x']).reshape(N)             # 'x'��a�ɑΉ�����B���O�����W���搔a�̃��X�g�B




# �T�|�[�g�x�N�g���̃C���f�b�N�X�̃��X�g���B
S = []
for i in range(len(a)):
	if a[i] < 0.00001: # �ق�0�Ȃ�T�|�[�g�x�N�g������Ȃ��B
		continue
	else:
		S += [i]

# w���v�Z�B
w = np.zeros(2) #�s���p�ӁB
for n in S:
	w += a[n]*t[n]*X[n]

# b���v�Z�B
sum = 0
for n in S: # �T�|�[�g�x�N�g���̐�����for���[�v�B
	temp = 0
	for m in S: # �T�|�[�g�x�N�g���̐�����for���[�v�B
		temp += a[m]*t[m]*kernel(X[n],X[m])
	sum += (t[n]-temp)
b = sum/len(S)




# �f�[�^���v���b�g�B
for p in (cls1+cls2):
	plot(p[0],p[1],'bo')

# �T�|�[�g�x�N�g����`��
for n in S:
	plot(X[n,0], X[n,1], 'ro')    

# ���ʊ֐���`��B
def f(x1, w, b):
    return - (w[0] / w[1]) * x1 - (b / w[1])

# ���ʋ��E��`��
x1 = np.linspace(-6, 6, 1000)
x2 = [f(x, w, b) for x in x1]
plot(x1, x2, 'g-')
    
xlim(0, 6)
ylim(0, 12)
show()

