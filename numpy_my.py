#coding:utf-8

import numpy
a = numpy.array([[2,2]])

#�x�N�g���̒���
length = numpy.linalg.norm(a)
#length=>2.8284271247461903

#�x�N�g���̐��K��
a / numpy.linalg.norm(a)
#=>array([[ 0.70710678,  0.70710678]])