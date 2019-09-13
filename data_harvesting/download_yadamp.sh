#!/bin/bash

mkdir YADAMP_pages
cd YADAMP_pages

#download 2525 webpages
for i in {1..2525}
do
    wget http://yadamp.unisa.it/showItem.aspx?yadampid=$i
    name=`echo $i|awk '{printf"peptide_%.4d.html\n", $1}'`
    mv showItem.aspx?yadampid=$i $name
done

cd ..
