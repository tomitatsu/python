#coding:utf-8

import numpy
a = numpy.array([[2,2]])

#ベクトルの長さ
length = numpy.linalg.norm(a)
#length=>2.8284271247461903

#ベクトルの正規化
a / numpy.linalg.norm(a)
#=>array([[ 0.70710678,  0.70710678]])