#coding:utf-8

# プロット用
from pylab import *


#################################
# ファイル読み込み
#################################
for line in open('data.csv', 'r'):
	itemList = line.strip().split('\t')
# この時点では文字列のリスト
#	print itemList
# 数値のリストに変換
	nums = [ int(item) for item in itemList ]
	print nums

#################################
# データをプロット。
#################################
for p in (nums):
	plot(p[0],p[1],'bo')


