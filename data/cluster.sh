#!/bin/bash

for((i=0;i<10;i++));
do
begin=$(expr $i \* 25000);
#echo $begin;
python cluster.py $begin;
done
