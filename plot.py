#coding:utf-8

# �v���b�g�p
from pylab import *


#################################
# �t�@�C���ǂݍ���
#################################
for line in open('data.csv', 'r'):
	itemList = line.strip().split('\t')
# ���̎��_�ł͕�����̃��X�g
#	print itemList
# ���l�̃��X�g�ɕϊ�
	nums = [ int(item) for item in itemList ]
	print nums

#################################
# �f�[�^���v���b�g�B
#################################
for p in (nums):
	plot(p[0],p[1],'bo')


